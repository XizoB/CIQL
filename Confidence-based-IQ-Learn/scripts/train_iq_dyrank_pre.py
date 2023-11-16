"""
Copyright 2022 Div Garg. All rights reserved.

Example training code for IQ-Learn which minimially modifies `train_rl.py`.
"""

import datetime
import os
import random
import time
from collections import deque                           # Replay Buffer 容器，类似于 list 的容器，可以快速的在队列头部和尾部添加、删除元素
from itertools import count                             # 创建复杂迭代器
import types                                            # 其中MethodType动态的给对象添加实例方法

import hydra                                            # 管理yaml config配置文件的，配合使用OmegaConf
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf             # to_yaml到无结构的文本信息str类型，通过换行符来分隔
from tensorboardX import SummaryWriter                  # tensorboard可视化训练

import pickle as pkl

from wrappers.atari_wrapper import LazyFrames
from make_envs import make_env
from dataset.memory import Memory
from agent import make_agent
from utils.utils import eval_mode, evaluate, soft_update, save, save_config, save_trajs, get_concat_samples, select_state, save_best
from utils.logger import Logger 
from ca_iq import ca_iq_update                          # confidence-aware 置信度感知学习

torch.set_num_threads(2)


def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() and cfg.device is None else "cpu"
    cfg.hydra_base_dir = os.getcwd()                    # 获得当前文件的根目录，最后获得的路径是在根目录下建立 /outputs/当前日期/此刻时间 文件夹
    print(OmegaConf.to_yaml(cfg))                       # 直接打印文本信息，注意除了上面两个信息不会在 /outputs/当前日期/此刻时间/.hydra/config.yaml文件中显示
    return cfg

def scale_state(scale,state):
    return scale * state

def cdynamic(args, expert_buffer, device):
    """
    ## object_obs 3类 共10个状态  ------  object-state (10)
    cube_pos (3)、cube_quat (4)、gripper_to_cube_pos (3)

    ## Panda 7类 共32个状态  ------  robot0_proprio-state (32)
    robot0_joint_pos_cos (7)、robot0_joint_pos_sin (7)、robot0_joint_vel (7)
    robot0_eef_pos (3,31-33)、robot0_eef_quat (4)
    robot0_gripper_qpos (2)、robot0_gripper_qvel (2)

    ## IIWA 7类 共40个状态  ------  robot0_proprio-state (37)
    robot0_joint_pos_cos (7)、robot0_joint_pos_sin (7)、robot0_joint_vel (7)
    robot0_eef_pos (3)、robot0_eef_quat (4)
    robot0_gripper_qpos (6)、robot0_gripper_qvel (6)
    """
    batch_size = expert_buffer.size()
    expert_batch, indexes = expert_buffer.get_samples(batch_size, True, device)  # 把 conf 传入 Memory 类中，以在里面全局使用
    batch_state, batch_next_state, _, batch_reward, _ = expert_batch

    conf = []

    pos = batch_state[:,31:34]
    n_pos = batch_next_state[:,31:34]
    obj_pos = batch_state[:,0:3]
    v_n = n_pos - pos
    v_o = obj_pos - pos

    # print(v_n)
    # print(v_o)
    angle_bound = args.train.boundary_angle
    for i in range(len(v_n)):
        cos_no = v_n[i].dot(v_o[i])/(np.linalg.norm(v_n[i])*np.linalg.norm(v_o[i]))
        angle_no = np.degrees(np.arccos(cos_no))/angle_bound
        if batch_reward[i] >= 0.45:
            conf.append(1.5)
            continue
        if batch_reward[i] >= 0.4 and batch_reward[i] < 0.45:
            conf.append(1)
            continue
        if angle_no <= 1:
            conf_value = (1/(1+np.exp(10*(angle_no-0.5))))
            # conf_value = (1-(angle_no/angle_bound)**2)/(1+((angle_no/angle_bound)**2))
            conf.append(conf_value.item())
        else:
            conf.append(1)
    return conf




# main函数里面的存储路径全在 /outputs/当前日期/此刻时间/ 目录下
@hydra.main(config_path="conf/iq", config_name="config")    # 配置的路径在"conf"，配置的文件名为"config"
def main(cfg: DictConfig):
    #------------ 设置参数 ------------#
    # 获取参数
    args = get_args(cfg)

    # 设置随机种子参数
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 设置 GPU 参数
    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # 设置实验参数
    REPLAY_MEMORY = int(args.env.replay_mem)                # Replay_Buffer 的大小
    INITIAL_MEMORY = int(args.env.initial_mem)              # 第一次更新前，Replay_Buffer 中需要提前存入的 Transition 片段数目
    HORIZON = int(args.env.horizon)                         # 智能体与环境交互的最长步数，限制每一个 EPOCH 的交互数
    # EPISODE_WINDOW = int(args.train.eps_window)           # 智能体训练过程中的奖励窗口大小
    LEARN_STEPS = int(args.train.learn_steps)               # 智能体更新次数
    ROUllOUT_LENGTH = int(args.train.roullout_length)       # 每一次更新循环前，智能体需要与环境交互的数目
    TRAIN_STEPS = int(args.train.train_steps)               # 每一次更新循环内，智能体的策略与价值网络的更新数目
    INITIAL_STATES = 128                                    # 采样数目，为计算专家初始状态 s0 值的分布


    #------------ 初始化 env 和 agent ------------#
    # 初始化训练环境 env
    env = make_env(args)                                    # 初始化训练环境
    if not args.env.ignore_done:
        args.env.ignore_done = True
    eval_env = make_env(args)                               # 初始化评估环境

    env.seed(args.seed)
    eval_env.seed(args.seed + 10)
    # env.action_space.seed(0)                                # env_action_space 随机采样种子

    # 初始化智能体 agent
    agent = make_agent(env, args)

    # 是否需要导入预训练的模型
    if args.pretrain:
        pretrain_path = hydra.utils.to_absolute_path(args.pretrain)
        print("=> loading pretrain '{}'".format(args.pretrain))
        agent.load(pretrain_path, f'_{args.env.name}')

    # 初始化专家与智能体的 Replay_buffer 并导入专家数据  
    expert_memory_replay = Memory(REPLAY_MEMORY//2, device, args.seed)          # 实例化Memory
    online_memory_replay = Memory(REPLAY_MEMORY//2, device, args.seed+1)        # 实例化Memory
    
    expert_memory_replay.load(hydra.utils.to_absolute_path(f'experts/{args.env.demo}'),
                              choice_trajs=args.expert.choice_trajs,
                              subopt_class_num=args.expert.subopt_class_num,
                              num_trajs=args.expert.demos,
                              sample_freq=args.expert.subsample_freq,
                              seed=args.seed + 42,
                              label_ratio=args.C_aware.label_ratio,
                              sparse_sample=args.C_aware.sparse_sample)
    print(f'--> Expert memory size: {expert_memory_replay.size()}')             # 打印Replay Buffer中专家数据的大小
    info_label = expert_memory_replay.info_trajs()
    save_trajs(info_label, output_dir='trajs')
    traj_label, length_label, reward_label = info_label

    ####### 设置置信度
    if args.C_aware.conf_learn=="IQ" or args.C_aware.conf_learn=="sail":
        conf = torch.ones(expert_memory_replay.size(), 1).to(device)            # tesnor [40*1000,1] 初始化专家数据的置信度
        print("lamb:", args.C_aware.lamb)
    else:
        conf = cdynamic(args, expert_memory_replay, device)
        conf = torch.Tensor(conf)
        print("最优先验概率:", conf.mean())

        if args.C_aware.conf_learn=="max_lamb":
            args.C_aware.lamb = max(conf.mean().item(), args.C_aware.lamb)
        if args.C_aware.conf_learn=="conf_expert":
            args.C_aware.lamb = conf.mean().item()

        print("boundary_angle:", args.train.boundary_angle)
        print("lamb:", args.C_aware.lamb)




    #------------ 设置训练日志 ------------#
    # 设置训练日志目录 Setup logging
    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, args.env.name, args.exp_name, ts_str)  # /outputs/当前日期/此刻时间/logs/CartPole-v1/2022-07-18_21-09-50
    writer = SummaryWriter(log_dir=log_dir)                                     # tensorboard可视化保存路径，在outputs目录下
    print(f'--> Saving logs at: {log_dir}')
    logger = Logger(args.log_dir, log_frequency=args.log_interval, writer=writer, save_tb=True, agent=args.agent.name)  # 实例化Logger类


    # 采样来自 env 的示例初始状态 Sample initial states from env
    # state_0 = [env.reset()] * INITIAL_STATES                                      # 统计初始状态分布 s_0 值的初始状态
    # if isinstance(state_0[0], LazyFrames):                                        # 判断两个类型是否相同
    #     state_0 = np.array(state_0) / 255.0
    # state_0 = torch.FloatTensor(np.array(state_0)).to(args.device)



    ####################################################################
    steps = 0                                                                     # 智能体与环境交互步数
    learn_steps = 0                                                               # 智能体学习更新步数
    best_succeed = 0
    begin_learn = False
    success_deque = deque(maxlen=10)
    # with open("/root/RoboLearn/Confidence-Aware-IQ-Learn/iq_learn/200_demo_14.43_scale.pkl", "rb") as p:
    #     scale = pkl.load(p)
    # print("scale:", scale)

    for epoch in count():                                                         # 创建无限迭代, 开始训练智能体
        
        state = env.reset()
        # state = scale_state(scale, state)
        episode_reward = 0
        done = False
        success = False

        start_time = time.time()                                                  # 记录一个 epoch 的开始时间

        # 每一个epoch/episode/trajectory 中 episode 的最大步数，避免让智能体一直探索或陷入死循环
        for episode_step in range(HORIZON):
            # env.render()

            #------ Step-1 ------ 
            # 智能体交互闭环： 与环境进行交互得到(s,a,r,s',done)
            if steps < args.num_seed_steps:                                     # 在训练前，带有随机种子随机选取动作存入 Replay Buffer
                action = env.action_space.sample()
            else:
                with eval_mode(agent):
                    action = agent.choose_action(state, sample=True)            # 智能体采取动作
            next_state, reward, done, _ = env.step(action)                      # 与环境进行交互
            # next_state = scale_state(scale, next_state)
            episode_reward += reward                                            # 叠加奖励
            steps += 1                                                          # 智能体与环境交互的总步长


            # 仅在一个 epoch/episode/horizon 到结束时候(done)才为1 到步数限制时为0
            # 在时间限制的环境中直接截断在固定交互步数中(1000, 允许无限引导 infinite bootstrap)
            done_no_lim = done
            if str(env.__class__.__name__).find('TimeLimit') >= 0 and episode_step + 1 == env._max_episode_steps:
                done_no_lim = 0
            online_memory_replay.add((state, next_state, action, reward, done_no_lim)) # 在online的replay_buffer中存入一组MDP元组


            #------ Step-2 ------ 
            # 智能体训练闭环
            if online_memory_replay.size() >= INITIAL_MEMORY and (steps-INITIAL_MEMORY) % ROUllOUT_LENGTH == 0:
                # 打印开始训练 Start learning
                if begin_learn is False:
                    print(f'Learn begins and online_memory_size is {online_memory_replay.size()}!')
                    begin_learn = True

                for _ in range(TRAIN_STEPS):
                    # --- 评估智能体， 按照评估间隙 eval_interval
                    if learn_steps % args.train.eval_interval == 0:
                        eval_returns, succceed = evaluate(agent, eval_env, args, logger, epoch, learn_steps)
                        save(agent, args, eval_returns, succceed, learn_steps, output_dir='results')
                        if succceed >= best_succeed:
                            best_succeed = succceed
                            save_best(agent, args, eval_returns, succceed, learn_steps, output_dir='results')
                        save_config(conf, learn_steps, output_dir='config')            # 保存置信度数据

                    # --- 判断是否结束训练
                    if learn_steps >= LEARN_STEPS:
                        print(f'Finished and online_memory_size is {online_memory_replay.size()}!')
                        # wandb.finish()
                        return                                                         # 结束主函数

                    ######
                    # --- IQ-Learn Modification 修改之前的算法
                    agent.iq_update = types.MethodType(iq_update, agent)               # MethodType动态的给对象添加实例方法
                    agent.ca_iq_update = types.MethodType(ca_iq_update, agent)         # MethodType动态的给对象添加实例方法
                    losses, conf = agent.iq_update(online_memory_replay, expert_memory_replay, conf, learn_steps)
                    learn_steps += 1                                                   # To prevent repeated eval at timestep 0
                    ######

                    # tensorboard可视化训练结果
                    if learn_steps % args.log_interval == 0:                           # 可视化显示的步长
                        for key, loss in losses.items():
                            writer.add_scalar(key, loss, global_step=learn_steps)

            if done:
                break
            state = next_state

            if reward == 1 and not success:
                success = True

        if success:
            success_deque.append(1)
        else:
            success_deque.append(0)

        # tensorboard 可视化
        # rewards_window.append(episode_reward)
        logger.log('train/episode_reward', episode_reward, epoch)
        logger.log('train/success_rate', success_deque.count(1)/10, epoch)
        logger.log('train/duration', time.time() - start_time, epoch)
        # logger.dump(learn_steps, save=begin_learn)
        # print('TRAIN\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, np.mean(rewards_window)))



def iq_update(self, policy_buffer, expert_buffer, conf, step):
    """
    IQ方法更新
    -> losses
    """
    loss_dict = {}

    #------------ 更新 Critic 网络 (IQ更新方法) ------------#
    self.conf = conf                                                                  # 在智能体类中共用conf

    # 智能体与专家MDP元组采样指定Batch大小的数据样本
    policy_batch, _ = policy_buffer.get_samples(self.batch_size, False, self.device)
    expert_batch, indexes = expert_buffer.get_samples(self.batch_size, False, self.device)  # 把 conf 传入 Memory 类中，以在里面全局使用
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
    expert_conf = self.conf[indexes]



    #------------ 更新 Actor 网络 (SAC更新方法, 延迟策略更新，更新频率低于Critic价值网络) ------------#
    if step % self.actor_update_frequency == 0:
        for _ in range(self.args.agent.num_actor_updates):
            actor_alpha_losses = self.update_actor_and_alpha_iq(batch, expert_conf)   # 与SAC中的策略网络与alpha值更新一样 集合中有五个列表

        loss_dict.update(actor_alpha_losses)


    return loss_dict, self.conf


if __name__ == "__main__":
    main()
