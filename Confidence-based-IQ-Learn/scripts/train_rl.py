import datetime
import os
import random
import time
import hydra
import numpy as np
import torch
import torch.nn.functional as F

from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter
from collections import deque
from itertools import count
from utils.logger import Logger
from make_envs import make_env
from dataset.memory import Memory
from agent import make_agent
from utils.utils import evaluate, eval_mode, save

torch.set_num_threads(2)


def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() and cfg.device is None else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    print(OmegaConf.to_yaml(cfg))
    return cfg

# main函数里面的存储路径全在 /outputs/当前日期/此刻时间/ 目录下
@hydra.main(config_path="conf/rl", config_name="config")
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

    # 初始化实验参数
    REPLAY_MEMORY = int(args.env.replay_mem)                # Replay_Buffer 的大小
    INITIAL_MEMORY = int(args.env.initial_mem)              # 第一次更新前，Replay_Buffer 中需要提前存入的 Transition 片段数目
    HORIZON = int(args.env.horizon)                         # 智能体与环境交互的最长步数，限制每一个 EPOCH 的交互数
    # EPISODE_WINDOW = int(args.train.eps_window)           # 智能体训练过程中的奖励窗口大小
    LEARN_STEPS = int(args.train.learn_steps)               # 智能体更新次数
    ROUllOUT_LENGTH = int(args.train.roullout_length)       # 每一次更新循环前，智能体需要与环境交互的数目
    TRAIN_STEPS = int(args.train.train_steps)               # 每一次更新循环内，智能体的策略与价值网络的更新数目

    #------------ 初始化 env 和 agent ------------#
    # 初始化训练环境 env
    env = make_env(args)                                    # 初始化训练环境
    eval_env = make_env(args)                               # 初始化评估环境

    env.seed(args.seed)
    eval_env.seed(args.seed + 10)
    # env.action_space.seed(0)                              # env_action_space 随机采样种子

    # 初始化智能体 agent
    agent = make_agent(env, args)

    # 是否需要导入预训练的模型
    if args.pretrain:
        pretrain_path = hydra.utils.to_absolute_path(args.pretrain)
        if os.path.exists(args.pretrain):
            print("=> loading pretrain '{}'".format(args.pretrain))
            agent.load(pretrain_path, f'_{args.env.name}')
        else:
            print("[Attention]: Did not find checkpoint {}".format(args.pretrain))

    # 初始化 Replay_Buffer
    memory_replay = Memory(REPLAY_MEMORY, args.seed)


    #------------ 设置训练日志 ------------#
    # 设置训练日志目录 Setup logging
    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, args.env.name, args.exp_name, ts_str)
    writer = SummaryWriter(log_dir=log_dir)
    print(f'--> Saving logs at: {log_dir}')
    logger = Logger(args.log_dir, log_frequency=args.log_interval, writer=writer, save_tb=True, agent=args.agent.name)

    # # 跟踪训练过程中的奖励 track mean reward and scores
    # rewards_window = deque(maxlen=EPISODE_WINDOW)  # last N rewards
    # best_eval_returns = -np.inf

    ####################################################################
    steps = 0                                               # 智能体与环境交互步数
    learn_steps = 0                                         # 智能体学习更新步数
    begin_learn = False

    for epoch in count():                                   # 创建无限迭代, 开始训练智能体

        state = env.reset()
        episode_reward = 0
        done = False

        start_time = time.time()                            # 记录一个epoch的开始时间

        # 每一个epoch/episode/trajectory 中 episode 的最大步数，避免让智能体一直探索或陷入死循环
        for episode_step in range(HORIZON):
            # env.render()

            #------ Step-1 ------ 
            # 智能体交互闭环： 与环境进行交互得到(s,a,r,s',done)
            if steps < args.num_seed_steps:
                action = env.action_space.sample()          # 在训练前，带有随机种子随机选取动作存入 Replay Buffer
            else:
                with eval_mode(agent):
                    action = agent.choose_action(state, sample=True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1
            
            # 仅在一个 epoch/episode/horizon 到结束时候(done)才为1 到步数限制时为0
            # 在时间限制的环境中直接截断在固定交互步数中(1000, 允许无限引导 infinite bootstrap)
            done_no_lim = done
            if str(env.__class__.__name__).find('TimeLimit') >= 0 and episode_step + 1 == env._max_episode_steps:
                done_no_lim = 0
            memory_replay.add((state, next_state, action, reward, done_no_lim))
            
            #------ Step-2 ------ 
            # 智能体训练闭环
            if memory_replay.size() >= INITIAL_MEMORY and (steps-INITIAL_MEMORY) % ROUllOUT_LENGTH == 0:
                # 打印开始训练 Start learning
                if begin_learn is False:
                    print(f'Learn begins and memory_replay_size is {memory_replay.size()}!')
                    begin_learn = True

                for _ in range(TRAIN_STEPS):
                    # --- 评估智能体， 按照评估间隙 eval_interval
                    if learn_steps % args.train.eval_interval == 0:
                        eval_returns = evaluate(agent, eval_env, args, logger, epoch, learn_steps)
                        learn_steps += 1                    # To prevent repeated eval at timestep 0
                        save(agent, args, eval_returns, learn_steps, output_dir='results')

                    # --- 判断是否结束训练
                    if learn_steps >= LEARN_STEPS:
                        print(f'Finished and memory_replay_size is {memory_replay.size()}!')
                        return                              # 结束主函数

                    # --- 训练智能体
                    losses = agent.update(memory_replay, logger, learn_steps)
                    learn_steps += 1

                    if learn_steps % args.log_interval == 0:
                        for key, loss in losses.items():
                            writer.add_scalar(key, loss, global_step=learn_steps)

            if done:
                break
            state = next_state

        # tensorboard 可视化
        # rewards_window.append(episode_reward)
        logger.log('train/episode', epoch, steps)
        logger.log('train/episode_reward', episode_reward, steps)
        logger.log('train/duration', time.time() - start_time, steps)
        # logger.dump(steps, save=begin_learn)
        # print('TRAIN\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, np.mean(rewards_window)))

    ####################################################################


if __name__ == "__main__":
    main()
