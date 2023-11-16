"""
A convenience script to playback random demonstrations from
a set of demonstrations stored in a hdf5 file.

Arguments:
    --folder (str): Path to demonstrations
    --use-actions (optional): If this flag is provided, the actions are played back
        through the MuJoCo simulator, instead of loading the simulator states
        one by one.
    --visualize-gripper (optional): If set, will visualize the gripper site

Example:
    $ python playback_demonstrations_from_hdf5.py --folder ../models/assets/demonstrations/SawyerPickPlace/
"""

import os
import h5py
import argparse
import json

import numpy as np
import torch

from torch import nn
from scipy.stats import spearmanr, pearsonr
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import build_mlp


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
            output_activation=nn.Sigmoid()

    ):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape + action_shape,
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation,
            output_activation=output_activation
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
        return self.net(torch.cat([states, actions], dim=-1))


def cdynamic(states, next_states, rewards):
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
    angle_bound = 60
    for i in range(len(v_n)):
        cos_no = v_n[i].dot(v_o[i])/(np.linalg.norm(v_n[i])*np.linalg.norm(v_o[i]))
        angle_no = np.degrees(np.arccos(cos_no))/angle_bound
        if rewards[i] >= 0.45:
            continue
        if rewards[i] >= 0.30 and rewards[i] < 0.45:
            conf.append(1)
            x_state.append(states[i][31])
            y_state.append(states[i][32])
            z_state.append(states[i][33])
            key.append(i)
            continue
        if angle_no <= 1:
            # print(angle_no)
            conf_value = (1/(1+np.exp(10*(angle_no-0.5))))
            # conf_value = (1-(angle_no)**2)/(1+((angle_no)**2))
            conf.append(conf_value)
            x_state.append(states[i][31])
            y_state.append(states[i][32])
            z_state.append(states[i][33])
            key.append(i)

    return x_state, y_state, z_state, conf, key


def main(args):
    #---- 导入参数 ---#
    demo_path = args.folder
    hdf5_path = os.path.join(demo_path, "robosuite_demo.hdf5")
    f = h5py.File(hdf5_path, "r")

    demos = list(f["data"].keys())  # 所有演示集的列表
    print("demos:", demos)

    states_shape = f["data/{}/states".format(demos[0])][()][1]
    states_shape = len(states_shape)
    actions_shape = np.array(f["data/{}/actions".format(demos[0])][()])[1]
    actions_shape = len(actions_shape)


    # 置信度函数
    if args.trexpath is not None:
        device = "cpu"
        confidence = ConfidenceFunction(
            state_shape=states_shape,
            action_shape=actions_shape,
            hidden_activation=nn.ReLU(inplace=True),
            output_activation=nn.Sigmoid()
        ).to(device)

        confidence_path = f'{args.trexpath}'
        confidence.load_state_dict(torch.load(confidence_path, map_location=device))




    count = 0
    HIRIZONS = []
    CONFIDENCE = []
    NORM_CONFIDENCE = []
    REWARDS = []
    DYNAMIC_CONFIDENCE = []
    GAMMA_REWARDS = []
    gamma = 0.99
    TARGET = []
    get_target = False
    N = 2
    WRONG_DEMOS = []
    NET_CONFIDENCE = []


    plt.style.use('seaborn')
    # ---------------- 末端位置图
    if args.pic >= 1:
        plt.figure(1)
        ax = plt.axes(projection='3d')
        ax.set_xlabel('X label') # 画出坐标轴
        ax.set_ylabel('Y label')
        ax.set_zlabel('Z label')
    
    if args.pic >= 2:
        plt.figure(2)
        ax2 = plt.axes(projection='3d')
        ax2.set_xlabel('X label') # 画出坐标轴
        ax2.set_ylabel('Y label')
        ax2.set_zlabel('Z label')

    STATES = np.array([])
    init = True
    for ep in demos:
        # load the flattened mujoco states
        states = np.array(f["data/{}/states".format(ep)][()], dtype=np.float32)
        actions = np.array(f["data/{}/actions".format(ep)][()], dtype=np.float32)
        next_states = np.array(f["data/{}/next_states".format(ep)][()], dtype=np.float32)
        rewards = f["data/{}/rewards".format(ep)][()]
        horizon = f["data/{}/horizon".format(ep)][()]


        # sc = np.ones(42)*10
        # states = sc * states
        # next_states = sc * next_states


        if init:
            STATES = states
            init =False
        np.vstack((STATES, states))

        # 末端位置
        # StackObstacle
        # x_state = states[:,51]
        # y_state = states[:,52]
        # z_state = states[:,53]

        # Stack
        # x_state = states[:,44]
        # y_state = states[:,45]
        # z_state = states[:,46]

        # Lift
        x_state = states[:,31]
        y_state = states[:,32]
        z_state = states[:,33]


        # # 目标物体
        # x_state = states[:,0]
        # y_state = states[:,1]
        # z_state = states[:,2]

        # 障碍物位置
        # x_state = states[:,14]
        # y_state = states[:,15]
        # z_state = states[:,16]

        # # 堆叠方块
        # x_state = states[:,7]
        # y_state = states[:,8]
        # z_state = states[:,9]
        
        # 动作
        x_action = actions[:,0]
        y_action = actions[:,1]
        z_action = actions[:,2]

        # ---------------- 存储
        # 存储 HIRIZONS
        HIRIZONS.append(horizon)

        # 存储 GAMMA_REWARDS
        G = 0
        for i in reversed(rewards):
            G = gamma * G + i
        GAMMA_REWARDS.append(G)

        # 存储 REWARDS
        R = int(np.array(rewards).sum())
        REWARDS.append(R)

        # 不同轨迹段的颜色标记
        get_target = False
        for i in range(N):
            if horizon >= 50*i and horizon < 50*(i+1):
                TARGET.append(i)
                get_target = True
                break
        if not get_target:
            TARGET.append(N)

        # 存储 NET_CONFIDENCE
        if args.trexpath is not None:
            c = confidence(states, actions).detach().numpy().mean()
            NET_CONFIDENCE.append(c)

        # 存储 DYNAMIC_CONFIDENCE
        x_state, y_state, z_state, conf, key = cdynamic(states, next_states, rewards)
        DYNAMIC_CONFIDENCE.append(np.array(conf).mean())

        # 画末端位置图
        clist = ["r", "g", "b"]
        get_target = False
        for i in range(N):
            if horizon <= 50:
                cvalue = clist[i]
                get_target = True
                break
        if not get_target:
            cvalue = clist[N]
        if (np.array(y_state) > 0.15).any():
            WRONG_DEMOS.append(ep + "_Horizon_" + str(horizon) + "_Reward_" + str(np.array(rewards).sum()))
        # ax.scatter3D(x_state,y_state,z_state,c=cvalue)  #绘制散点图
        ax.scatter3D(x_state,y_state,z_state,c=conf,cmap="bwr")  #绘制散点图
        ax.scatter3D(states[1][0],states[1][1],states[1][2],marker="*",c=cvalue,s=200)  #绘制散点图
        # ax.scatter3D(states[5][0],states[5][1],states[5][2],marker="*",c=cvalue,s=200)  #绘制散点图
        ax.scatter3D(states[key[len(key)-1]][0],states[key[len(key)-1]][1],states[key[len(key)-1]][2],marker="*",c=cvalue,s=200)  #绘制散点图

        # 画动作图
        action_bound = np.hstack((x_action,y_action,z_action))
        if (action_bound > 1).any():
            WRONG_DEMOS.append(ep + "_Horizon_" + str(horizon) + "_Reward_" + str(np.array(rewards).sum()))
        if args.pic >= 2:
            ax2.scatter3D(x_action,y_action,z_action)  #绘制散点图


        if count >= 200:
            break
        count += 1

        print("Ep {}\t Horizon: {}\t Reward:{:.2f}\t".format(ep, horizon,np.array(rewards).sum()))



    max_states = np.max(STATES,axis=0)
    min_states = np.min(STATES,axis=0)
    scale = []
    print("State_Max :{}".format(max_states))
    print("State_Min :{}".format(min_states))


    for i in range(len(min_states)):
        s = 1
        done = False
        while not done:
            if np.abs(max_states[i]*s) >= 0.1:
                scale.append(s)
                break
            elif np.abs(min_states[i]*s) >= 0.1:
                scale.append(s)
                break
            else:
                s *= 10
    
    print("scale:", scale)
    print("State_Max :{}".format(max_states*np.array(scale)))
    print("State_Min :{}".format(min_states*np.array(scale)))
    
    print("demos numbers:{}".format(len(demos)))
    print("WRONG_DEMOS:", WRONG_DEMOS)





    # 轨迹分布图
    if args.pic >= 3:
        plt.figure(3)
        plt.subplot(2,2,1)
        plt.hist(HIRIZONS,200)
        plt.title("demos_horizon")
        plt.xlabel("horizon")
        plt.ylabel("nums")

        # 奖励分布图
        plt.subplot(2,2,2)
        plt.hist(REWARDS,200)
        plt.title("demos_rewards")
        plt.xlabel("rewards")
        plt.ylabel("nums")
        plt.subplot(2,2,3)


    # -------------- 长度 xx 关系分布图
    fig, ax = plt.subplots(2,2,figsize=(12, 7))

    # 长度 奖励
    ax[0,0].scatter(HIRIZONS, REWARDS, c=TARGET, cmap="rainbow", s=50)
    ax[0,0].set(title="REWARDS vs HIRIZONS",
       xlabel="HIRIZONS",
       ylabel="REWARDS")
    
    z = np.polyfit(HIRIZONS, REWARDS, 1) # 用4次多项式拟合
    p1 = np.poly1d(z)
    print(p1)
    yvals = p1(HIRIZONS)
    ax[0,0].plot(HIRIZONS, yvals)

    # 长度 GAMMA奖励
    ax[0,1].scatter(HIRIZONS, GAMMA_REWARDS, c=TARGET, cmap="rainbow", s=50)
    ax[0,1].set(title="GAMMA_REWARDS vs HIRIZONS",
       xlabel="HIRIZONS",
       ylabel="GAMMA_REWARDS")
    

    HIRIZONS = np.array(HIRIZONS)
    REWARDS = np.array(REWARDS)
    conf_norm = REWARDS/HIRIZONS
    for i in range(len(HIRIZONS)):
        conf_value = (conf_norm[i]-conf_norm.min())/(conf_norm.max()-conf_norm.min())
        conf_length = (HIRIZONS.max()-HIRIZONS[i])/(HIRIZONS.max()-HIRIZONS.min())
        conf_value = conf_value * conf_length
        CONFIDENCE.append(conf_value)

    CONFIDENCE = np.array(CONFIDENCE)
    for i in range(len(CONFIDENCE)):
        conf_value = (CONFIDENCE[i]-CONFIDENCE.min())/(CONFIDENCE.max()-CONFIDENCE.min())
        NORM_CONFIDENCE.append(conf_value)


    # 长度 (奖励/长度 * 长度)置信度
    ax[1,0].scatter(HIRIZONS, CONFIDENCE, c=TARGET, cmap="rainbow", s=50)
    ax[1,0].set(title="CONFIDENCE vs HIRIZONS",
       xlabel="HIRIZONS",
       ylabel="CONFIDENCE")


    # 长度 dynamic置信度
    if args.pic >= 1:
        ax[1,1].scatter(HIRIZONS, DYNAMIC_CONFIDENCE, s=50)
        ax[1,1].set(title="HIRIZONS vs DYNAMIC_CONFIDENCE",
        xlabel="HIRIZONS",
        ylabel="DYNAMIC_CONFIDENCE")

    # ax[1,1].scatter(HIRIZONS, NET_CONFIDENCE, c=TARGET, cmap="rainbow", s=50)
    # ax[1,1].set(title="NET_CONFIDENCE vs HIRIZONS",
    #    xlabel="HIRIZONS",
    #    ylabel="NET_CONFIDENCE")


    print("============= REWARDS =============")
    print(f'Spearman correlation: {spearmanr(HIRIZONS, REWARDS)}')
    print(f'Pearson correlation: {pearsonr(HIRIZONS, REWARDS)}')


    plt.show()
    f.close()



if __name__ == "__main__":
    #---- 输入参数 ---#
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="path_to_assets_dir/demonstrations/YOUR_DEMONSTRATIO"
    )
    parser.add_argument(
        "--trexpath",
        type=str,
        default=None,
        help="path_to_assets_dir/demonstrations/YOUR_DEMONSTRATIO"
    )
    parser.add_argument(
        "--pic",
        type=int,
        default=2,
        help="path_to_assets_dir/demonstrations/YOUR_DEMONSTRATIO"
    )


    #---- 回放演示数据 ---#
    args = parser.parse_args()
    main(args)

