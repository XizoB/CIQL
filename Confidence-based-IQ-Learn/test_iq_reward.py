from itertools import count

import hydra
import torch
import numpy as np
import os
from omegaconf import DictConfig, OmegaConf
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import wandb


from dataset.memory import Memory
from make_envs import make_env
from agent import make_agent
# from utils.utils import evaluate

def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() and cfg.device == "gpu" else "cpu"
    print(OmegaConf.to_yaml(cfg))
    return cfg


def cdynamic(states, next_states, rewards, irl_reward):
    """
    ## object_obs 3类 共10个状态  ------  object-state (10)
    cube_pos (3)、cube_quat (4)、gripper_to_cube_pos (3)

    ## Panda 7类 共32个状态  ------  robot0_proprio-state (32)
    robot0_joint_pos_cos (7)、robot0_joint_pos_sin (7)、robot0_joint_vel (7)
    robot0_eef_pos (3,31-33)、robot0_eef_quat (4)
    robot0_gripper_qpos (2)、robot0_gripper_qvel (2)
    """
    x_state = []
    y_state = []
    z_state = []
    key = []
    conf = []

    pos = states[:,31:34]
    n_pos = next_states[:,31:34]
    obj_pos = states[:,0:3]
    v_n = n_pos - pos
    v_o = obj_pos - pos

    # print(v_n)
    # print(v_o)
    angle_bound = 30
    delete_key = []
    for i in range(len(v_n)):
        cos_no = v_n[i].dot(v_o[i])/(np.linalg.norm(v_n[i])*np.linalg.norm(v_o[i]))
        angle_no = np.degrees(np.arccos(cos_no))/angle_bound
        if rewards[i] >= 0.45:
            conf.append(1.5)
            delete_key.append(i)
            continue
        if rewards[i] >= 0.4 and rewards[i] < 0.45:
            conf.append(1)
            delete_key.append(i)
            continue
        if angle_no <= 1:
            # print(angle_no)
            conf_value = (1/(1+np.exp(10*(angle_no-0.5))))
            # conf_value = (1-(angle_no)**2)/(1+((angle_no)**2))
            conf.append(conf_value)
            continue
        else:
            conf.append(0)
            # delete_key.append(i)

    delete_key_true = False
    if delete_key_true:
        irl_reward = np.delete(irl_reward, delete_key)

    return conf, irl_reward

@hydra.main(config_path="conf/iq", config_name="config")
def main(cfg: DictConfig):
    #------------ 初始化环境参数设置 ------------#
    
    args = get_args(cfg)                                # 获取参数
    device = torch.device(args.device)

    REPLAY_MEMORY = int(args.env.replay_mem)                # Replay_Buffer 的大小


    env = make_env(args, monitor=False)                 # 初始化 env 环境
    agent = make_agent(env, args)                       # 初始化 agent 智能体

    # 导入 agent 智能体网络
    policy_file = f'results/{args.method.type}.para'
    if args.eval.policy:
        policy_file = f'{args.eval.policy}'
    print(f'Loading policy from: {policy_file}')

    if args.eval.transfer:
        agent.load(hydra.utils.to_absolute_path(policy_file), f'_{args.eval.expert_env}')
    else:
        agent.load(hydra.utils.to_absolute_path(policy_file), f'_{args.env.name}')


    expert_memory_replay = Memory(REPLAY_MEMORY//2, device, args.seed)          # 实例化Memory
    
    expert_memory_replay.load(hydra.utils.to_absolute_path(f'experts/{args.env.demo}'),
                              choice_trajs=args.expert.choice_trajs,
                              subopt_class_num=args.expert.subopt_class_num,
                              num_trajs=args.expert.demos,
                              sample_freq=args.expert.subsample_freq,
                              seed=args.seed + 42,
                              label_ratio=args.C_aware.label_ratio,
                              sparse_sample=args.C_aware.sparse_sample)
    print(f'--> Expert memory size: {expert_memory_replay.size()}')             # 打印Replay Buffer中专家数据的大小




    # #------------  评估智能体 --> 回报 更新步数 ------------#
    # eval_returns, eval_timesteps = evaluate(agent, env, num_episodes=args.eval.eps)
    # print(f'Avg. eval returns: {np.mean(eval_returns)}, timesteps: {np.mean(eval_timesteps)}')
    # if args.eval_only:
    #     exit()


    # #------------  测量相关性 ------------#
    measure_correlations(args, agent, env, expert_memory_replay, log=True)



def measure_correlations(args, agent, env, expert_memory_replay, log=False):
    GAMMA = args.gamma

    states_traj, next_states_traj, actions_traj, reward_traj, _ = expert_memory_replay.sample_all_expert_traj()

    print("states_traj:", states_traj[0].shape)


    GAMMA_REWARDS = []
    GAMMA_REWARDS_MEAN = []
    REWARDS = []
    REWARDS_MEAN = []
    HORIZONS = []
    DYNAMIC_CONF = []
    STATES_CONF = []
    LABEL = []

    initial_irl_reward = True

    for i in range(len(states_traj)):

        ######
        # Get sqil reward
        with torch.no_grad():
            q = agent.infer_q(states_traj[i], actions_traj[i])
            next_v = agent.infer_v(next_states_traj[i])
            done = torch.zeros(len(states_traj[i])).view(-1,1)
            for j in range(5):
                done[-1-j] = 1
            y = (1 - done) * GAMMA * next_v.view(-1,1)
            irl_reward = (q - y)  # 恢复奖励函数
            # irl_reward = torch.sigmoid(irl_reward)
            # irl_reward = torch.tanh(irl_reward)


        # 存储 GAMMA_REWARDS
        conf, irl_reward = cdynamic(states_traj[i], next_states_traj[i], reward_traj[i], irl_reward.numpy())

        G = 0
        for j in reversed(irl_reward):
            G = GAMMA * G + j
    
        if len(states_traj[i]) <= 200:
            LABEL.append("r")
        else:
            LABEL.append("blue")

        REWARDS.append(irl_reward.sum().item())
        REWARDS_MEAN.append(irl_reward.sum().item()/len(states_traj[i]))
        GAMMA_REWARDS.append(G.item())
        GAMMA_REWARDS_MEAN.append(G.item()/len(states_traj[i]))
        HORIZONS.append(len(states_traj[i]))

        if initial_irl_reward:
            IRL_REWARD = irl_reward.squeeze()
            initial_irl_reward = False
        else:
            IRL_REWARD = np.concatenate((IRL_REWARD, irl_reward.squeeze()),axis=0)
        

        DYNAMIC_CONF.append(np.mean(conf))
        STATES_CONF.extend(conf)
        print("Traj:{}\t Hotizon:{}\t Irl_reward:{}\t".format(i+1, len(states_traj[i]), irl_reward.sum().item()))



    # mask = [sum(x) < -5 for x in env_rewards]  # skip outliers
    # env_rewards = [env_rewards[i] for i in range(len(env_rewards)) if mask[i]]
    # learnt_rewards = [learnt_rewards[i] for i in range(len(learnt_rewards)) if mask[i]]


    # 奖励函数推理结果保存路径 vis/{args.env.name}/correlation
    # plt.show()
    savedir = hydra.utils.to_absolute_path(f'vis/{args.env.name}/correlation')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # 画出并保存每一个episode轨迹的环境真实奖励的回报与算法恢复的奖励的回报
    sns.set()
    # fig = plt.figure(dpi=150)
    fig, ax = plt.subplots(2,2,figsize=(10, 6))
    ax[0,0].scatter(HORIZONS, GAMMA_REWARDS, s=10, alpha=0.8)
    ax[0,0].set(xlabel="HIRIZONS", ylabel="GAMMA_REWARDS")

    ax[1,0].scatter(HORIZONS, GAMMA_REWARDS_MEAN, s=10, alpha=0.8)
    ax[1,0].set(xlabel="HIRIZONS", ylabel="GAMMA_REWARDS_MEAN")

    ax[0,1].scatter(HORIZONS, REWARDS, s=10, alpha=0.8)
    ax[0,1].set(xlabel="HIRIZONS", ylabel="REWARDS")

    ax[1,1].scatter(HORIZONS, REWARDS_MEAN, s=10, alpha=0.8)
    ax[1,1].set(xlabel="HIRIZONS", ylabel="REWARDS_MEAN")

    plt.savefig(savedir + '/%s.png' % 'REWARD')
    plt.close()



    # 置信度
    sns.set()
    # fig = plt.figure(dpi=150)
    fig1, ax1 = plt.subplots(2,2,figsize=(10, 6))
    ax1[0,0].scatter(HORIZONS, DYNAMIC_CONF, s=10, alpha=0.8)
    ax1[0,0].set(xlabel="HIRIZONS", ylabel="DYNAMIC_CONF")

    ax1[1,0].scatter(IRL_REWARD, STATES_CONF, s=10, alpha=0.8)
    ax1[1,0].set(xlabel="REWARDS", ylabel="DYNAMIC_CONF")

    ax1[0,1].scatter(HORIZONS, GAMMA_REWARDS, s=10, alpha=0.8)
    ax1[0,1].set(xlabel="HIRIZONS", ylabel="GAMMA_REWARDS")

    ax1[1,1].scatter(GAMMA_REWARDS, DYNAMIC_CONF, s=10, alpha=0.8)
    ax1[1,1].set(xlabel="GAMMA_REWARDS", ylabel="DYNAMIC_CONF")

    # plt.xlim((0,500))
    plt.ylim((0,1))
    plt.savefig(savedir + '/%s.png' % 'DYNAMIC_CONF')
    plt.close()

    print("============= correlation =============")
    # print(f"{len(GAMMA_REWARDS)} <==> {len(DYNAMIC_CONF)}")
    # print(f'Spearman correlation: {spearmanr(GAMMA_REWARDS, DYNAMIC_CONF)}')
    # print(f'Pearson correlation: {pearsonr(GAMMA_REWARDS, DYNAMIC_CONF)}')



    # 画出并保存每一个episode轨迹的环境真实奖励的回报与算法恢复的奖励的回报
    sns.set()
    # fig = plt.figure(dpi=150)
    fig, ax2 = plt.subplots(1,1,figsize=(10, 6))
    ax2.scatter(HORIZONS, GAMMA_REWARDS, c=LABEL, s=30, alpha=0.8)
    ax2.set(xlabel="Horizons", ylabel="Rewards")
    plt.savefig(savedir + '/%s.png' % 'GAMMA_REWARDS')
    plt.close()




    # sns.set()
    # plt.figure(dpi=150)
    # plt.scatter(HORIZONS, GAMMA_REWARDS_MEAN, s=10, alpha=0.8)
    # plt.xlabel('HORIZONS')
    # plt.ylabel('GAMMA_REWARDS_MEAN')

    # plt.savefig(savedir + '/%s.png' % 'Episode rewards')
    # plt.close()

def eps(rewards):
    """
    求出每一个epoch中的回报(奖励总和)
    -> [sum(epoch_1), sum(epoch_2), ...]
    """
    return [sum(x) for x in rewards]


def part_eps(rewards):
    """
    沿列表列出累积的的回报(奖励总和)
    -> [epoch[0,0+1,0+1+2,...], epoch[...], ...]
    """
    return [np.cumsum(x) for x in rewards]


def evaluate(actor, env, num_episodes=10, vis=True):
    """Evaluates the policy.
    Args:
      actor: A policy to evaluate.
      env: Environment to evaluate the policy on.
      num_episodes: A number of episodes to average the policy on.
    Returns:
      Averaged reward and a total number of steps.
    """
    total_timesteps = []
    total_returns = []

    while len(total_returns) < num_episodes:
        state = env.reset()
        done = False

        while not done:
            action = actor.choose_action(state, sample=False)
            next_state, reward, done, info = env.step(action)
            state = next_state

            if 'episode' in info.keys():
                total_returns.append(info['episode']['r'])
                total_timesteps.append(info['episode']['l'])

    return total_returns, total_timesteps



if __name__ == '__main__':
    main()
