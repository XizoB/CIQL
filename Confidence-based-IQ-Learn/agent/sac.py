import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import hydra

from utils.utils import soft_update


class SAC(object):
    def __init__(self, obs_dim, action_dim, action_range, batch_size, args):
        self.gamma = args.gamma
        self.batch_size = batch_size
        self.action_range = action_range  # 环境的动作范围
        self.device = torch.device(args.device)  # 使得智能体的策略网络与价值网络传入GPU
        self.args = args
        agent_cfg = args.agent  # SAC算法参数

        self.critic_tau = agent_cfg.critic_tau  # SAC的价值网络的软更新tau参数
        self.learn_temp = agent_cfg.learn_temp  # SAC学习温度系数
        self.actor_update_frequency = agent_cfg.actor_update_frequency  # 策略网络更新频率
        self.critic_target_update_frequency = agent_cfg.critic_target_update_frequency  # 目标价值网络更新频率

        # 价值网络与策略网络初始化
        self.critic_1 = hydra.utils.instantiate(agent_cfg.critic_cfg, args=args).to(self.device)  # 初始化第一个Q价值网络
        self.critic_2 = hydra.utils.instantiate(agent_cfg.critic_cfg, args=args).to(self.device)  # 初始化第二个Q价值网络
        self.critic_target_1 = hydra.utils.instantiate(agent_cfg.critic_cfg, args=args).to(self.device)  # 初始化第一个目标Q价值网络
        self.critic_target_2 = hydra.utils.instantiate(agent_cfg.critic_cfg, args=args).to(self.device)  # 初始化第二个目标Q价值网络
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())  # 第一个Q价值网络的结构参数 --> 第一个目标Q价值网络结构参数
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())  # 第一个Q价值网络的结构参数 --> 第二个目标Q价值网络结构参数

        self.detached_critic = hydra.utils.instantiate(agent_cfg.detached_critic_cfg, args=args).to(self.device)  # 初始化伪更新价值网络
       
        self.actor = hydra.utils.instantiate(agent_cfg.actor_cfg).to(self.device)  # 初始化策略网络结构参数

        self.log_alpha = torch.tensor(np.log(agent_cfg.init_temp)).to(self.device)  # 初始化自动调节正则化参数alpha
        self.log_alpha.requires_grad = True
        # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
        self.target_entropy = -action_dim  # 初始化目标熵

        # optimizers 优化器
        self.actor_optimizer = Adam(self.actor.parameters(), lr=agent_cfg.actor_lr, betas=agent_cfg.actor_betas)
        self.critic_1_optimizer = Adam(self.critic_1.parameters(), lr=agent_cfg.critic_lr, betas=agent_cfg.critic_betas)
        self.critic_2_optimizer = Adam(self.critic_2.parameters(), lr=agent_cfg.critic_lr, betas=agent_cfg.critic_betas)
        self.log_alpha_optimizer = Adam([self.log_alpha], lr=agent_cfg.alpha_lr, betas=agent_cfg.alpha_betas)
        
        # 训练
        self.train()
        self.critic_target_1.train()
        self.critic_target_2.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic_1.train(training)
        self.critic_2.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def critic_net(self):
        return self.critic

    @property
    def critic_target_net(self):
        return self.critic_target

    def choose_action(self, state, sample=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        dist = self.actor(state)
        action = dist.sample() if sample else dist.mean
        # assert action.ndim == 2 and action.shape[0] == 1
        return action.detach().cpu().numpy()[0]

    def update(self, replay_buffer, logger, step):
        """
        RL 智能体更新
        """
        # 从 Replay_Buffer 中采样 batch_size 大小的 transitions 片段
        policy_batch, _ = replay_buffer.get_samples(self.batch_size)
        obs, next_obs, action, reward, done = policy_batch

        # 更新Q价值网络
        losses = self.update_critic(obs, action, reward, next_obs, done)

        # 更新策略网络
        if step % self.actor_update_frequency == 0:
            actor_alpha_losses = self.update_actor_and_alpha(obs)
            losses.update(actor_alpha_losses)

        # Q价值网络软更新
        if step % self.critic_target_update_frequency == 0:
            soft_update(self.critic_1, self.critic_target_1, self.critic_tau)
            soft_update(self.critic_2, self.critic_target_2, self.critic_tau)

        return losses

    def update_critic(self, obs, action, reward, next_obs, done):
        next_action, log_prob, _ = self.actor.sample(next_obs) # 根据当前策略网络计算下一个时刻的动作与策略熵

        # 计算目标Q值 target_Q
        target_Q_1 = self.critic_target_1(next_obs, next_action) # 根据目标Q1网络计算下一时刻的Q1值
        target_Q_2 = self.critic_target_2(next_obs, next_action) # 根据目标Q1网络计算下一时刻的Q1值
        target_V = torch.min(target_Q_1, target_Q_2) - self.alpha.detach() * log_prob # 下一时刻的Q值在Q1与Q2中选取最小的+策略熵
        target_Q = reward + (1 - done) * self.gamma * target_V # Bellman残差（TD误差）优势函数

        # 更新两个Q网络 get current Q estimates 
        current_Q1 = self.critic_1(obs, action)
        current_Q2 = self.critic_2(obs, action)
        critic_1_loss = (F.mse_loss(current_Q1, target_Q.detach())).mean() # Q1网络的损失
        critic_2_loss = (F.mse_loss(current_Q2, target_Q.detach())).mean() # Q2网络的损失

        # 梯度下降更新
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        return {
            'train/critic_1_loss': critic_1_loss.item(),
            'train/critic_2_loss': critic_2_loss.item()
        }

    def update_actor_and_alpha(self, obs):
        action, log_prob, _ = self.actor.sample(obs)  # 根据当前策略网络计算此时刻的动作与策略熵

        # 此刻状态和动作的两个Q价值网络的值
        actor_Q_1 = self.critic_1(obs, action)
        actor_Q_2 = self.critic_2(obs, action)

        # 策略网络的损失 即KL散度
        actor_loss = (self.alpha.detach() * log_prob - torch.min(actor_Q_1, actor_Q_2)).mean() 

        # 梯度下降更新 optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        losses = {
            'train/actor_loss': actor_loss.item(),
            'train/target_entropy': self.target_entropy,
            'train/actor_entropy': -log_prob.mean().item()
        } # losses集合存入 策略网络损失函数 目标熵 策略熵


        # 更新alpha值，自动熵调节法
        if self.learn_temp:
            self.log_alpha_optimizer.zero_grad()
            # 根据策略熵来定义alpha的损失函数
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()

            losses.update({
                'train/alpha_loss': alpha_loss.item(),
                'train/alpha_value': self.alpha.item()
            }) # losses集合存入 alpha_loss损失函数与损失值

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        return losses


    def update_iq(self, replay_buffer, logger, step):
        obs, next_obs, action, reward, done = replay_buffer.get_samples(self.batch_size, self.device)

        losses = self.update_critic_iq(obs, action, reward, next_obs, done, logger, step)
        actor_alpha_losses = self.update_actor_and_alpha(obs, logger, step)

        if step % self.actor_update_frequency == 0:
            losses.update(actor_alpha_losses)

        if step % self.critic_target_update_frequency == 0:
            soft_update(self.critic, self.critic_target, self.critic_tau)

        return losses


    def update_critic_iq(self, obs, action, reward, next_obs, done, logger, step):

        with torch.no_grad():
            next_action, log_prob, _ = self.actor.sample(next_obs)

            target_Q = self.critic_target(next_obs, next_action)
            target_V = target_Q - self.alpha.detach() * log_prob
            target_Q = reward + (1 - done) * self.gamma * target_V

        # get current Q estimates
        current_Q = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # self.critic.log(logger, step)
        return {'loss/critic': critic_loss.item()}

    def update_actor_and_alpha_iq(self, batch, expert_conf):
        lamb = self.args.C_aware.lamb
        obs, is_expert, is_agent = batch[0], batch[-2], batch[-1]

        action, log_prob, _ = self.actor.sample(obs)  # 根据当前策略网络计算此时刻的动作与策略熵
        actor_Q_1 = self.critic_1(obs, action)  # 此刻状态和动作的两个Q价值网络的值

        # 策略网络的损失 即KL散度
        if  self.args.offline:
            # 离线更新策略网络，只使用专家的状态
            actor_loss = (self.alpha.detach() * log_prob - actor_Q_1)[is_expert].mul(expert_conf/lamb).mean()
        elif self.args.online:
            actor_loss = (self.alpha.detach() * log_prob - actor_Q_1)[is_agent].mean()
        else:
            # 在线更新策略网络，使用专家的状态与智能体的状态一起更新
            actor_loss = (((self.alpha.detach() * log_prob - actor_Q_1)[is_expert].mul(expert_conf/lamb)
                            +(self.alpha.detach() * log_prob - actor_Q_1)[is_agent])/2).mean()

        # 梯度下降更新 optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        losses = {
            'train/actor_loss': actor_loss.item(),
            'train/target_entropy': self.target_entropy,
            'train/actor_entropy': -log_prob.mean().item()
        } # losses集合存入 策略网络损失函数 目标熵 策略熵


        # 更新alpha值，自动熵调节法
        if self.learn_temp:
            self.log_alpha_optimizer.zero_grad()
            # 根据策略熵来定义alpha的损失函数
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()

            losses.update({
                'train/alpha_loss': alpha_loss.item(),
                'train/alpha_value': self.alpha.item()
            }) # losses集合存入 alpha_loss损失函数与损失值

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        return losses



    # Save model parameters
    def save(self, path, suffix=""):
        actor_path = f"{path}{suffix}_actor"
        critic_1_path = f"{path}{suffix}_critic_1"
        critic_2_path = f"{path}{suffix}_critic_2"

        # print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic_1.state_dict(), critic_1_path)
        torch.save(self.critic_2.state_dict(), critic_2_path)

    # Load model parameters
    def load(self, path, suffix=""):
        actor_path = f'{path}/{self.args.agent.name}_{self.args.method.type}{suffix}_actor'
        critic_1_path = f'{path}/{self.args.agent.name}_{self.args.method.type}{suffix}_critic_1'
        critic_2_path = f'{path}/{self.args.agent.name}_{self.args.method.type}{suffix}_critic_2'

        print('Loading models from {} and {}, {}'.format(actor_path, critic_1_path, critic_2_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        if critic_1_path and critic_2_path is not None:
            self.critic_1.load_state_dict(torch.load(critic_1_path, map_location=self.device))
            self.critic_2.load_state_dict(torch.load(critic_2_path, map_location=self.device))

    def getV(self, obs):
        with torch.no_grad():
            action, log_prob, _ = self.actor.sample(obs) # 非常耗时间
        current_Q_1 = self.critic_1(obs, action)
        current_V_1 = current_Q_1 - self.alpha.detach() * log_prob
        return current_V_1, log_prob

    def get_targetV(self, obs):
        action, log_prob, _ = self.actor.sample(obs)
        target_Q_1 = self.critic_target_1(obs, action)
        target_V_1 = target_Q_1 - self.alpha.detach() * log_prob
        return target_V_1

    def infer_q(self, state, action):
        q = self.critic_1(state, action)
        return q.squeeze(0)

    def infer_v(self, state):
        obs = state
        with torch.no_grad():
            action, log_prob, _ = self.actor.sample(obs) # 非常耗时间
        current_Q_1 = self.critic_1(obs, action)
        current_V_1 = current_Q_1 - self.alpha.detach() * log_prob
        v = current_V_1.squeeze()
        return v

    def detached_getV(self, obs):
        with torch.no_grad():
            action, log_prob, _ = self.actor.sample(obs) # 非常耗时间
        current_Q = self.detached_critic(obs, action)
        current_V = current_Q - self.alpha.detach() * log_prob
        return current_V

    def detached_infer_q(self, state, action):
        q = self.detached_critic(state, action)
        return q.squeeze()

    def detached_infer_v(self, state):
        obs = state
        with torch.no_grad():
            action, log_prob, _ = self.actor.sample(obs) # 非常耗时间
        current_Q_1 = self.critic_1(obs, action)
        current_V_1 = current_Q_1 - self.alpha.detach() * log_prob
        v = current_V_1.squeeze()
        return v

    def sample_actions(self, obs, num_actions):
        """For CQL style training."""
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(
            obs.shape[0] * num_actions, obs.shape[1])
        action, log_prob, _ = self.actor.sample(obs_temp)
        return action, log_prob.view(obs.shape[0], num_actions, 1)

    def _get_tensor_values(self, obs, actions, network=None):
        """For CQL style training."""
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(
            obs.shape[0] * num_repeat, obs.shape[1])
        preds = network(obs_temp, actions)
        preds = preds.view(obs.shape[0], num_repeat, 1)
        return preds

    def cqlV(self, obs, network, num_random=10):
        """For CQL style training."""
        # importance sampled version
        action, log_prob = self.sample_actions(obs, num_random)
        current_Q = self._get_tensor_values(obs, action, network)

        random_action = torch.FloatTensor(
            obs.shape[0] * num_random, action.shape[-1]).uniform_(-1, 1).to(self.device)

        random_density = np.log(0.5 ** action.shape[-1])
        rand_Q = self._get_tensor_values(obs, random_action, network)
        alpha = self.alpha.detach()

        cat_Q = torch.cat(
            [rand_Q - alpha * random_density, current_Q - alpha * log_prob.detach()], 1
        )
        cql_V = torch.logsumexp(cat_Q / alpha, dim=1).mean() * alpha
        return cql_V
