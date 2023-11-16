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
from torch import nn
from torch.optim import Adam

from torch.nn.utils.convert_parameters import parameters_to_vector
from tqdm import tqdm
from wrappers.atari_wrapper import LazyFrames
from make_envs import make_env
from dataset.memory import Memory
from agent import make_agent
from utils.trex_utils import build_mlp
from utils.utils import eval_mode, evaluate, soft_update, save, save_config, save_trajs
from utils.logger import Logger 
from ca_iq import ca_iq_update                          # confidence-aware 置信度感知学习

torch.set_num_threads(2)


def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() and cfg.device == "gpu" else "cpu"
    cfg.hydra_base_dir = os.getcwd()                    # 获得当前文件的根目录，最后获得的路径是在根目录下建立 /outputs/当前日期/此刻时间 文件夹
    print(OmegaConf.to_yaml(cfg))                       # 直接打印文本信息，注意除了上面两个信息不会在 /outputs/当前日期/此刻时间/.hydra/config.yaml文件中显示
    return cfg


class ConfidenceFunction(nn.Module):
    """
    Value function that takes s-a pairs as input

    Parameters
    ----------
    state_shape: np.array
        shape of the state space
    action_shape: np.array
        shape of the action space
    hidden_units: tuple
        hidden units of the value function
    hidden_activation: nn.Module
        hidden activation of the value function
    """
    def __init__(
            self,
            state_shape: int,
            action_shape: int,
            hidden_units: tuple = (256, 256),
            hidden_activation=nn.ReLU(),
    ):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape,
            output_dim=action_shape,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
        )


    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Return values of the s-a pairs

        Parameters
        ----------
        states: torch.Tensor
            input states
        actions: torch.Tensor
            actions corresponding to the states

        Returns
        -------
        values: torch.Tensor
            values of the s-a pairs
        """
        return self.net(torch.cat([states], dim=-1))


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
    env.seed(args.seed)

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
    traj_reward_label, length_label, reward_label = info_label



    #------------ 设置训练日志 ------------#
    # 设置训练日志目录 Setup logging
    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, args.env.name, args.exp_name, ts_str)  # /outputs/当前日期/此刻时间/logs/CartPole-v1/2022-07-18_21-09-50
    writer = SummaryWriter(log_dir=log_dir)                                     # tensorboard可视化保存路径，在outputs目录下
    print(f'--> Saving logs at: {log_dir}')
    logger = Logger(args.log_dir, log_frequency=args.log_interval, writer=writer, save_tb=True, agent=args.agent.name)  # 实例化Logger类




    ####################################################################
    epoch_reward = 40000
    l2_ratio = 0.01
    lr_reward = 1e-5
    steps_exp = 10
    learning_steps_reward = 0
    batch_size = args.train.batch

    # 置信度函数
    confidence = ConfidenceFunction(
        state_shape=env.observation_space.shape[0],
        action_shape=env.action_space.shape[0],
        hidden_activation=nn.ReLU(inplace=True),
        # output_activation=nn.Sigmoid()
    ).to(device)

    # 置信度优化器
    optim_confidence = Adam(confidence.parameters(), lr=lr_reward)


    expert_batch, indexes = expert_memory_replay.get_samples(1000*batch_size, device)
    batch_state, batch_next_state, batch_action, batch_reward, batch_done = expert_batch
    print("Batch大小:",batch_state.shape)
    print("Batch大小:",batch_action.shape)

    w = confidence(batch_state, batch_action)
    print("最大:", w.max())
    print("最小:", w.min())


    if args.trexeval is None:
        """# train reward function"""
        print("Training reward function")
        for t in tqdm(range(epoch_reward)):

            states_x_exp = torch.Tensor([]).to(device)
            states_y_exp = torch.Tensor([]).to(device)
            actions_x_exp = torch.Tensor([]).to(device)
            actions_y_exp = torch.Tensor([]).to(device)
            base_label = torch.Tensor([]).to(device)

            for i in range(batch_size):

                label_states_traj, label_actions_traj = expert_memory_replay.sample_expert_traj()

                # print("shape.states:", type(label_states_traj))
                # print("shape.actions:", type(label_actions_traj))

                # print("shape.states:", type(label_states_traj[0]))
                # print("shape.actions:", type(label_actions_traj[0]))

                # print("shape.states:", type(label_states_traj[0][0]))
                # print("shape.actions:", type(label_actions_traj[0][0]))

                # print("shape.states:", label_states_traj[0].shape)
                # print("shape.actions:", label_actions_traj[0].shape)

                # print("shape.states:", label_states_traj[0][0].shape)
                # print("shape.actions:", label_actions_traj[0][0].shape)


                traj_x_states = label_states_traj[0]
                traj_y_states = label_states_traj[1]
                traj_x_actions = label_actions_traj[0]
                traj_y_actions = label_actions_traj[1]

                x_ptr = np.random.randint(traj_x_states.shape[0] - steps_exp)
                y_ptr = np.random.randint(traj_y_states.shape[0] - steps_exp)

                states_x_exp = torch.cat((states_x_exp, traj_x_states[x_ptr:x_ptr + steps_exp].unsqueeze(0)), dim=0)
                states_y_exp = torch.cat((states_y_exp, traj_y_states[y_ptr:y_ptr + steps_exp].unsqueeze(0)), dim=0)
                actions_x_exp = torch.cat((actions_x_exp, traj_x_actions[x_ptr:x_ptr + steps_exp].unsqueeze(0)), dim=0)
                actions_y_exp = torch.cat((actions_y_exp, traj_y_actions[y_ptr:y_ptr + steps_exp].unsqueeze(0)), dim=0)

                if len(traj_x_states) < len(traj_y_states):
                    base_label = torch.cat((base_label, torch.zeros(1).to(device)), dim=0)
                else:
                    base_label = torch.cat((base_label, torch.ones(1).to(device)), dim=0)

            # train reward function
            logits_x_exp = confidence(states_x_exp, actions_x_exp).sum(1)
            logits_y_exp = confidence(states_y_exp, actions_y_exp).sum(1)

            # reward function is to tell which trajectory is better by giving rewards based on states
            base_label = base_label.long()
            logits_xy_exp = torch.cat((logits_x_exp, logits_y_exp), dim=1)
            loss_cal = torch.nn.CrossEntropyLoss()
            loss_cmp = loss_cal(logits_xy_exp, base_label)
            loss_l2 = l2_ratio * parameters_to_vector(confidence.parameters()).norm() ** 2
            loss_reward = loss_cmp + loss_l2

            optim_confidence.zero_grad()
            loss_reward.backward()
            optim_confidence.step()


            if learning_steps_reward % 1000 == 0:
                tqdm.write(f'step: {learning_steps_reward}, loss: {loss_reward.item():.3f}')
                output_dir = f'results'
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
                torch.save(confidence.state_dict(), f'{output_dir}/{learning_steps_reward}_loss_{loss_reward.item():.3f}')
            learning_steps_reward += 1
        

    else:

        # Load model parameters
        confidence_path = f'{args.trexeval}'
        confidence.load_state_dict(torch.load(confidence_path, map_location=device))

    # j = 0
    # for i in range(len(length_label)):
    #     j += length_label[i]
    #     print("奖励函数{}--{}:".format(reward_label[j-1],reward_label[j]))


    



    ####################################################################


    


if __name__ == "__main__":
    main()
