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
import pickle
import datetime
import robosuite

import numpy as np
import robosuite as suite

from robosuite.controllers import load_controller_config
from robosuite.wrappers import GymWrapper
from robosuite.utils.mjcf_utils import postprocess_model_xml
from collections import defaultdict



def get_data_stats(failed_ep, rewards, lengths):
    print("成功的轨迹:\n")
    print("rewards: {:.2f} +/- {:.2f}".format(rewards.mean(), rewards.std()))
    print("len: {:.2f} +/- {:.2f}".format(lengths.mean(), lengths.std()))
    print("失败的轨迹:",failed_ep)



def get_env_config(raw_directory):
    """
    环境 env 参数配置
    """

    ep_directory_path = os.path.join(raw_directory, os.listdir(raw_directory)[0])   # 子演示文件夹路径
    hdf5_path = os.path.join(ep_directory_path, "demo.hdf5")        # demo 路径

    # JOINT_VELOCITY JOINT_POSITION OSC_POSE
    f = h5py.File(hdf5_path, "r")
    config = json.loads(f["data"].attrs["env_args"])
    config["controller_configs"]["type"] = args.controller
    controller_config = load_controller_config(default_controller=config["controller_configs"]["type"])
    config["controller_configs"] = controller_config
    config["env_kwargs"]["controller_configs"] = controller_config
    env_args = json.dumps(config, indent=4)  # 获取对控制器配置的引用并将其转换为json编码的字符串
    del config["env_kwargs"]  # 移除不必要的keys
    del config["type"]
    f.close()

    return config, env_args


def save_as_pkl(args, ep_directory, ep_directory_path, raw_directory, expert_trajs, new_demo_name):
    # 重命名
    new_name_path = os.path.join(raw_directory, new_demo_name)
    os.rename(ep_directory_path, new_name_path)
    print("{} 重命名为 ==> {}".format(ep_directory, new_demo_name))

    # 创建保存 pkl 的文件夹
    new_pkl_dir = os.path.join(args.folder, args.controller, "mix_pkl")
    if not os.path.exists(new_pkl_dir):
        os.makedirs(new_pkl_dir)

    # 保存 pkl 文件
    save_dir = os.path.join(new_pkl_dir, f'{new_demo_name}_num_{len(expert_trajs["lengths"])}_noice_{args.noice}.pkl')
    with open(save_dir, 'wb') as savef:
        pickle.dump(expert_trajs, savef)


def save_as_hdf5(args, env_args, ep, f, new_demo_name, traj, is_robosuite):
    # 创建保存 hdf5 的文件夹
    new_hdf5_dir = os.path.join(args.folder, args.controller, "raw", f"{new_demo_name}")
    if not os.path.exists(new_hdf5_dir):
        os.makedirs(new_hdf5_dir)

    if not is_robosuite:
        hdf5_path = os.path.join(new_hdf5_dir, "demo.hdf5")        # demo 路径
    else:
        hdf5_path = os.path.join(new_hdf5_dir, "robosuite_demo.hdf5")        # demo 路径


    # 保存 demo
    save_f = h5py.File(hdf5_path, "w")
    grp_demo = save_f.create_group("data")  # 在一个 "data" 组的属性中存储数据
    now = datetime.datetime.now()

    # --- 写入数据集属性
    grp_demo.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp_demo.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp_demo.attrs["repository_version"] = suite.__version__
    grp_demo.attrs["env"] = f["data"].attrs["env"]
    grp_demo.attrs["env_args"] = env_args

    # --- 存入数据
    states, next_states, actions, rewards, dones, demo_states, demo_next_states = zip(*traj)

    if not is_robosuite:
        states = np.vstack(demo_states)
        next_states = np.vstack(demo_next_states)
    else:
        states = np.vstack(states)
        next_states = np.vstack(next_states)       
    actions = np.vstack(actions)
    rewards = np.vstack(rewards).squeeze()
    dones = np.vstack(dones).squeeze()
    horizon = len(states)

    ep_data_grp = grp_demo.create_group("{}".format(ep))
    ep_data_grp.create_dataset("states", data=states)
    ep_data_grp.create_dataset("actions", data=actions)
    ep_data_grp.create_dataset("next_states", data=next_states)
    ep_data_grp.create_dataset("rewards", data=rewards)
    ep_data_grp.create_dataset("dones", data=dones)
    ep_data_grp.create_dataset("horizon", data=horizon)
    ep_data_grp.attrs["model_file"] = f["data/{}".format(ep)].attrs["model_file"]

    save_f.close()





def main(args):
    # ------ 导入参数 ------#
    # 所有演示数据所在 merge_directory 文件夹
    raw_directory = os.path.join(args.folder, "Raw")
    print("Raw_demo_path:", raw_directory)


    # ------ 初始化环境 ------ #
    config, env_args = get_env_config(raw_directory)
    env = robosuite.make(**config)
    env = GymWrapper(env)


    # 循环文件夹中的每一个演示文件夹
    saved_eps = 0
    expert_lengths = []
    expert_rewards = []
    failed_ep = []
    for ep_directory in os.listdir(raw_directory):
        ep_directory_path = os.path.join(raw_directory, ep_directory)   # 子演示文件夹路径
        hdf5_path = os.path.join(ep_directory_path, "demo.hdf5")        # demo 路径
        hdf5_robosuite_path = os.path.join(ep_directory_path, "robosuite_demo.hdf5")        # demo 路径


        f = h5py.File(hdf5_path, "r")
        f_robosuite = h5py.File(hdf5_robosuite_path, "r")


        demos = list(f["data"].keys())  # 所有演示集的列表
        print("演示 demos:", demos)


        # ------ 开始复现仿真
        # 保存轨迹的容器
        expert_trajs = defaultdict(list)
        try_times = 0       # 每个演示轨迹复现尝试的次数
        while True:
            try_times += 1
            for ep in demos:
                print("Playing back random episode... (press ESC to quit)")

                # ------ 使用演示数据集 data 属性中读取模型xml
                model_xml = f["data/{}".format(ep)].attrs["model_file"]


                # ------ 使用演示数据集中的初始状态初始化
                env.reset()
                xml = postprocess_model_xml(model_xml)
                env.reset_from_xml_string(xml)
                env.sim.reset()
                env.viewer.set_camera(0)


                # ------ 导入演示数据集中的初始状态
                # load the flattened mujoco states, next_states, actions
                states = f["data/{}/states".format(ep)][()]
                n_states = f["data/{}/next_states".format(ep)][()]
                actions = np.array(f["data/{}/actions".format(ep)][()])
                num_actions = actions.shape[0]
                print("轨迹长度: {}\t 控制器: {}\t 噪声: {}".format(num_actions, args.controller, args.noice))
                # load the initial state
                env.sim.set_state_from_flattened(states[0] + np.random.uniform(-args.noice,args.noice,len(states[0])))
                env.sim.forward()


                # ------ 使用演示数据集中的动作仿真
                episode_reward = 0
                traj = []
                init_states = True
                for i, action in enumerate(actions):
                    # 判断机械臂的控制器
                    if args.controller == "OSC_POSE":
                        action_noice = np.random.uniform(-args.noice,args.noice,7)
                        next_state, reward, done, info = env.step(action + action_noice)
                        demo_next_state = env.sim.get_state().flatten()
                    else:
                        action_noice = np.random.uniform(-args.noice,args.noice,7)
                        action = np.append(n_states[i,35:42]+action_noice,action[-1]) * 2.22
                        action = np.clip(action,-1,1)
                        next_state, reward, done, info = env.step(action)
                        demo_next_state = env.sim.get_state().flatten()


                    
                    if not init_states:
                        traj.append((state, next_state, action, reward, done, demo_state, demo_next_state))
                    init_states = False
            
                    episode_reward += reward
                    state = next_state
                    demo_state = demo_next_state
                    env.render()

                    if i < num_actions - 1:
                        # 判断仿真的状态是否与记录的状态一样 recorded states
                        state_playback = env.sim.get_state().flatten()
                        if not np.all(np.equal(states[i + 1], state_playback)):
                            err = np.linalg.norm(states[i + 1] - state_playback)
                            print(f"[warning] playback diverged by {err:.2f} for ep {ep} at step {i}")

                if reward == 0:
                    ifsave = True
                    saved_eps += 1
                    states, next_states, actions, rewards, dones, _, _ = zip(*traj)
                    new_demo_name = f"{args.controller}_{len(traj)}_{round(episode_reward,3)}"

                    expert_trajs["states"].append(states)
                    expert_trajs["next_states"].append(next_states)
                    expert_trajs["actions"].append(actions)
                    expert_trajs["rewards"].append(rewards)
                    expert_trajs["dones"].append(dones)
                    expert_trajs["lengths"].append(len(traj))
                    expert_lengths.append(len(traj))
                    expert_rewards.append(episode_reward)
                    print('总保存轨迹: {}\t 决策步长: {}\t奖励为: {:.2f}\tEp: {}\t'.format(saved_eps, len(traj), episode_reward, ep))
                    
                    save_as_hdf5(args, env_args, ep, f, new_demo_name, traj,is_robosuite=False)
                    save_as_hdf5(args, env_args, ep, f_robosuite, new_demo_name, traj, is_robosuite=True)

                else:
                    ifsave = False
                    failed_ep.append(f"{ep_directory}_{ep}")
                    print("尝试次数: {}".format(try_times))
                    print('跳过该轨迹: {}\t Ep:{}\t奖励为: {:.2f}\t'.format(f"{len(traj)}_{round(episode_reward,3)}",ep, episode_reward))
            
            if len(expert_trajs["lengths"]) >= args.nums or try_times >= args.try_times:
                break

        
        if ifsave:
            save_as_pkl(args, ep_directory, ep_directory_path, raw_directory, expert_trajs, new_demo_name)
            print("----------------------------------------")
            print("")

        f.close()
    
    
    # 最后数据日志输出
    get_data_stats(failed_ep, np.array(expert_rewards), np.array(expert_lengths))


    exit()



if __name__ == "__main__":
    #---- 输入参数 ---#
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="path_to_assets_dir/demonstrations/YOUR_DEMONSTRATIO"
    )
    # 开环回放存储的数据集操作，而不是从模拟状态加载
    parser.add_argument(
        "--controller",
        default="OSC_POSE",
        type=str,
        help="JOINT_VELOCITY JOINT_POSITION OSC_POSE",
    )
    parser.add_argument(
        "--noice",
        default='0',
        type=float,
        help="noice",
    )
    parser.add_argument(
        "--nums",
        default='1',
        type=int,
        help="nums",
    )
    parser.add_argument(
        "--try-times",
        default='1',
        type=int,
        help="nums",
    )

    #---- 回放演示数据 ---#
    args = parser.parse_args()
    main(args)

