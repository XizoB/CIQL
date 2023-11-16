"""
Copyright 2022 Div Garg. All rights reserved.

Standalone IQ-Learn algorithm. See LICENSE for licensing terms.
"""
from turtle import shape
import torch
import torch.nn.functional as F
import itertools
import time
import math

from torch import nn
from torch.autograd import Variable
from torch.nn.utils.convert_parameters import parameters_to_vector

from collections import Counter
from utils.utils import average_dicts, get_concat_samples



def get_soft_weight(self, expert_traj_batch, step):
    conf_loss_dict = {}
    GAMMA = self.args.gamma
    initial_step = self.args.train.initial_step
    learn_steps = self.args.train.learn_steps
    exp_n = self.args.C_aware.exp_n

    states_traj, next_states_traj, actions_traj, rewards_traj, done_traj = expert_traj_batch # tensor([[轨迹1[1000个state[],..[]]],...,[轨迹n]])

    if self.args.C_aware.Qvalue:
        reward = self.detached_infer_q(states_traj, actions_traj)
    else:
        q = self.detached_infer_q(states_traj, actions_traj)
        next_v = self.detached_infer_v(next_states_traj)
        y = (1 - done_traj.squeeze()) * GAMMA * next_v
        reward = q - y
        # reward = 5 * torch.tanh(q - y)

    k_min = torch.min(reward.detach())
    k_mid = torch.median(reward.detach())

    self.k = k_mid + ((step - initial_step) / learn_steps) ** exp_n * (k_mid - k_min)
    if self.args.C_aware.sail_update =="soft": # soft_update
        expert_conf = 1. / (1. + math.e ** (self.k - reward.detach()))
    elif self.args.C_aware.sail_update =="hard": # hard_update
        expert_conf = (reward.detach() >= self.k).float()

    conf_loss_dict['conf_loss/k'] = self.k
    conf_loss_dict['conf_loss/irl_reward_mean'] = reward.mean()
    conf_loss_dict['conf_loss/irl_reward_min'] = reward.min()
    conf_loss_dict['conf_loss/irl_reward_mid'] = reward.median()
    conf_loss_dict['conf_loss/irl_reward_max'] = reward.max()
    conf_loss_dict['conf_loss/conf_sum'] = expert_conf.sum()


    return self.k, expert_conf.unsqueeze(1), conf_loss_dict


def ca_iq_update(self, policy_buffer, expert_buffer, step):
    loss_dict = {}

    # 智能体与专家MDP元组采样指定Batch大小的数据样本
    policy_batch, _ = policy_buffer.get_samples(self.batch_size)
    expert_batch, indexes = expert_buffer.get_samples(self.batch_size)  # 把 conf 传入 Memory 类中，以在里面全局使用
    # expert_traj_batch = expert_buffer.sample_expert_traj(self.args.C_aware.traj_batch_size)  # 从专家样本中采样轨迹Batch

    # 仅使用观察值对 IL 使用策略操作而不是专家操作
    # if self.args.only_expert_states:
    #     policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
    #     expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch[0:-2] # 一个batch的专家样本
    #     # Use policy actions instead of experts actions for IL with only observations
    #     expert_batch = expert_obs, expert_next_obs, policy_action, expert_reward, expert_done

    # 智能体与专家的MDP元组数据合并与解压
    batch = get_concat_samples(policy_batch, expert_batch[0:5], self.args)


    # ------------------- 第一步，更新置信度
    if self.args.C_aware.conf_learn=="sail":
        if step >= self.args.train.initial_step:
            self.k, expert_conf, conf_loss_dict = get_soft_weight(self, expert_batch, step)
            loss_dict.update(conf_loss_dict)
            self.conf[indexes] = Variable(expert_conf)
        else:
            # 定义智能体类的参数，以求在全局计算
            expert_conf = self.conf[indexes]
    else:
        expert_conf = self.conf[indexes]


    # ------------------- 第二步，更新Q值函数的theta
    critic_1_loss, critic_loss_dict = update_critic_grad(self, batch, Variable(expert_conf))
    loss_dict.update(critic_loss_dict)

    self.critic_1_optimizer.zero_grad()
    critic_1_loss.backward()
    self.critic_1_optimizer.step()
    loss_dict['train/critic_1_loss'] = critic_1_loss

    return loss_dict, batch, expert_conf



def update_critic_grad(self, batch, expert_conf):
    obs, next_obs, action = batch[0:3]

    # 获得当前观测状态的 V 价值与下一观测状态的 V‘ 价值，比较耗费时间
    # 判断是否使用target网络
    current_V, log_prob = self.getV(obs)  # 根据当前观测状态获得V_current值
    with torch.no_grad():
        next_V = self.get_targetV(next_obs)
    current_Q = self.critic_1(obs, action)

    critic_loss, loss_dict = ca_iq_loss(self, current_Q, current_V, log_prob, next_V, batch, expert_conf)

    return critic_loss, loss_dict



def ca_iq_loss(self, current_Q, current_v, log_prob, next_v, batch, expert_conf):
    ######
    # IQ-Learn minimal implementation with X^2 divergence (~15 lines)
    args = self.args
    gamma = self.gamma
    lamb = args.C_aware.lamb
    expert_conf = expert_conf.squeeze()

    loss_dict = {}
    obs, next_obs, action, env_reward, done, is_expert, is_agent = batch

    y = (1 - done) * gamma * next_v

    if args.C_aware.conf_learn=="IQ":
        # ########### ---0 IQ-Learning
        # 1st term of loss: -E_(ρ_expert)[Q(s, a) - γV(s')]
        reward = (current_Q - y)[is_expert]
        loss = -(reward).mean()
        loss_dict['iq_loss/softq_loss'] = loss.item()  # loss_dict集合中存入softq_loss值
        
        # 2nd term for our loss (use expert and policy states): E_(ρ)[V(s,a) - γV(s')]
        if args.online:
            value_loss = ((current_v - y)[is_agent]).mean()
        else:
            value_loss = (((current_v - y)[is_expert]+(current_v - y)[is_agent])/2).mean()
        loss += value_loss
        loss_dict['iq_loss/value_loss'] = value_loss.item()

        # Use χ2 divergence (adds a extra term to the loss)
        # calculate the regularization term for IQ loss using expert and policy states (works online)
        reward = torch.cat(((current_Q - y)[is_expert],(current_Q - y)[is_agent])) # tensor.Size(256)
        chi2_loss = 1/(4 * args.method.alpha) * (reward**2).mean()
        loss += chi2_loss
        loss_dict['iq_loss/chi2_loss'] = chi2_loss.item()
        loss_dict['iq_loss/total_loss'] = loss.item()


    elif args.C_aware.conf_learn=="conf_expert":
        ########### ---1 Confidence Expert Learning
        # 1st term of loss: -beta * E_(ρ_expert)[Q(s, a) - γV(s')]
        reward = (current_Q - y)[is_expert].mul(expert_conf/lamb)
        loss = -(reward).mean()
        loss_dict['iq_loss/softq_loss'] = loss.item()  # loss_dict集合中存入softq_loss值
        
        # 2nd term for our loss (use expert and policy states): E_(ρ)[Q(s,a) - γV(s')]
        value_loss = (((current_v - y)[is_expert].mul(expert_conf/lamb)+(current_v - y)[is_agent])/2).mean()
        loss += value_loss
        loss_dict['iq_loss/value_loss'] = value_loss.item()

        # Use χ2 divergence (adds a extra term to the loss)
        # calculate the regularization term for IQ loss using expert and policy states (works online)
        reward = torch.cat(((current_Q - y)[is_expert].mul(expert_conf/lamb),(current_Q - y)[is_agent])) # tensor.Size(256)
        chi2_loss = 1/(4 * args.method.alpha) * (reward**2).mean()
        loss += chi2_loss
        loss_dict['iq_loss/chi2_loss'] = chi2_loss.item()
        loss_dict['iq_loss/total_loss'] = loss.item()


    elif args.C_aware.conf_learn=="max_lamb":
        # ########### ---3 IC-IQL max_lamb
        # 1st term of loss: -E_(ρ_expert)[Q(s, a) - γV(s')]
        reward = (current_Q - y)[is_expert]
        loss = -(reward).mean()
        loss_dict['iq_loss/softq_loss'] = loss.item()  # loss_dict集合中存入softq_loss值
        
        # 2nd term for our loss (use expert and policy states): E_(ρ)[Q(s,a) - γV(s')]
        value_loss = (((current_v - y)[is_expert].mul(expert_conf)+(current_v - y)[is_agent])/2).mean()
        log_prob_loss = ((log_prob[is_expert].mul(expert_conf)+log_prob[is_agent])/2).mean()
        loss += lamb * value_loss + (1-lamb) * log_prob_loss
        loss_dict['iq_loss/value_loss'] = value_loss.item()
        loss_dict['iq_loss/log_prob_loss'] = log_prob_loss.item()

        # Use χ2 divergence (adds a extra term to the loss)
        # calculate the regularization term for IQ loss using expert and policy states (works online)
        reward = current_Q - y # tensor.Size(256,1)
        chi2_loss = 1/(4 * args.method.alpha) * (reward**2).mean()
        loss += chi2_loss
        loss_dict['iq_loss/chi2_loss'] = chi2_loss.item()

        reward = (current_Q - y)[is_expert].mul(1-expert_conf)
        label_value_loss = ((1-lamb)/(1-expert_conf.mean())*reward).mean()
        loss += label_value_loss
        loss_dict['iq_loss/label_value_loss'] = label_value_loss.item()  # loss_dict集合中存入softq_loss值
        loss_dict['iq_loss/total_loss'] = loss.item()

    return loss, loss_dict

