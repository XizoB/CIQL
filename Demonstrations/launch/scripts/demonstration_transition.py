import os
import json
import pickle
import h5py
import numpy as np
import argparse

from collections import defaultdict


"""
python /root/RoboLearn/Test/Test_benchmark/Launch_run/demonstration_transition.py --dataset_path /root/RoboLearn/Download/Data/lift/ph --output_dir /root/RoboLearn/Test/Test_launch/Data_train

"""

def select_state(state):
    """
    ## object_obs 3类 共10个状态  ------  object-state (10)
    cube_pos (3)、cube_quat (4)、gripper_to_cube_pos (3)

    ## Panda 7类 共32个状态  ------  robot0_proprio-state (32)
    robot0_joint_pos_cos (7)、robot0_joint_pos_sin (7)、robot0_joint_vel (7)
    robot0_eef_pos (3,31-33)、robot0_eef_quat (4)
    robot0_gripper_qpos (2)、robot0_gripper_qvel (2)
    """
    key = [0,1,2,3,4,5,6,31,32,33,34,35,36,37,38,39]
    print("状态长度:", len(key))
    return state[:,key]


def scale_filter(states):
    max_states = np.max(states, axis=0)
    min_states = np.min(states, axis=0)
    scale = []
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
    return np.array(scale)



def scale_select(states):
    max_states = np.max(states, axis=0)
    min_states = np.min(states, axis=0)
    scale = []
    for i in range(len(min_states)):
        if i <= 1:
            scale.append(2)
            continue
        elif i >= 3 and i <= 4:
            scale.append(2)
            continue
        elif i == 8:
            scale.append(2)
            continue
        elif i == 32:
            scale.append(2)
            continue
        elif i == 37:
            scale.append(2)
            continue
        else:
            scale.append(1)
    return np.array(scale)

def test_robosuite_demonstrations(f, demos):

    traj_horizons = []

    init_action_box = True
    for ep in demos:

        states = f["data/{}/states".format(ep)][()]
        actions = f["data/{}/actions".format(ep)][()]
        next_states = f["data/{}/next_states".format(ep)][()]
        rewards = f["data/{}/rewards".format(ep)][()]
        dones = f["data/{}/dones".format(ep)][()]
        horizon = f["data/{}/horizon".format(ep)][()]

        # 判断数据长度是否相等, states 与 next_states 是否相等
        assert (len(states) == len(actions) 
                and len(states) == len(rewards) 
                and len(states) == len(dones)
                and np.allclose(next_states[:-1],states[1:])
                and len(states) == horizon
                )
        
        if np.min(actions) < -1 or np.max(actions) > 1:
            print("wrongs demos:",ep)

        if init_action_box:
            init_action_box = False
            action_min = np.min(actions,axis=0)
            action_max = np.max(actions,axis=0)
        else:
            action_min = np.vstack((action_min,np.min(actions,axis=0)))
            action_min = np.min(action_min,axis=0)
            action_max = np.vstack((action_max,np.max(actions,axis=0)))
            action_max = np.max(action_max,axis=0)

        traj_horizons.append(horizon)

    # traj_horizons = np.array(traj_horizons)

    # 查看 robosuite demonstrations 数据
    print("")
    print("==== Robosuite Demonstrations statistics ====")
    print("")
    print("total transitions: {}".format(np.sum(traj_horizons)))
    print("total trajectories: {}".format(traj_horizons))
    print("traj length mean: {}".format(np.mean(traj_horizons)))
    print("traj length std: {}".format(np.std(traj_horizons)))
    print("traj length min: {}".format(np.min(traj_horizons)))
    print("traj length max: {}".format(np.max(traj_horizons)))

    print("动作最小: {}".format(action_min))
    print("动作最大: {}".format(action_max))

    print("states nums: {}".format(states[0].shape))
    print("actions nums: {}".format(actions[0].shape))



def test_iqlearn_demonstrations(iqlearn_dataset_pkl, dataset_output_dir):

    # 查看 IQ-Learn 数据
    print("")
    print("==== IQ-Learn Data statistics ====")
    print("")
    # obs_success = []
    # for i in range(len(states)):
    #     obs_success.append(np.allclose(states[i][1:], next_states[i][:-1]))
    # print("states equal next_states: {}".format(all(obs_success)))


    f = open(iqlearn_dataset_pkl, "rb")
    info = pickle.load(f)

    print("type(info):", type(info))
    print("len(info['states']):", len(info['states']))
    print("len(info['states'][0]):", len(info['states'][0]))
    print("len(info['next_states'][0][0]):", len(info['next_states'][0][0]))

    print("")
    info = dict(info)
    print("info.keys():", info.keys())
    # print(info["dones"][0])
    print("type(info[states]):", type(info["states"]))
    print("type(info[states][0]):", type(info["states"][0]))
    print("type(info[states][0][0]):", type(info["states"][0][0]))
    print("info[states][0][0].shape:", info["states"][0][8].shape)

    print("")
    print("type(info[actions]):", type(info["actions"]))
    print("type(info[actions][0]):", type(info["actions"][0]))
    print("type(info[actions][0][0]):", type(info["actions"][0][0]))
    print("info[actions][0][0].shape:", info["actions"][0][10].shape)

    print("")
    print("type(info[dones][0]):", type(info["dones"][0]))
    print("type(info[dones][0][0]):", type(info["dones"][0][0]))

    print("")
    # print("rewards:", info["rewards"])
    print("type(info[rewards][0]):", type(info["rewards"][0]))
    print("type(info[rewards][0][0]):", type(info["rewards"][0][0]))

    print("")
    print("type(info[lengths]):", type(info["lengths"]))
    print("type(info[lengths][0]):", type(info["lengths"][0]))

    # np.savetxt("{}/iq_rewards_pkl.txt".format(dataset_output_dir), info["rewards"][0])

    mean_return = 0
    return_list = []
    n = len(info['states'])
    for i in range(n):
        epoch_return = np.array(info["rewards"][i])
        return_list.append(epoch_return.sum())
        mean_return += epoch_return.sum()/n
    print("each_return:", return_list)
    print("mean_return:", mean_return)


def failed_state(failed, state):
    state = state[0:-failed]
    return state


def mix_demonstration(args):

    #---- 导入数据 ----#
    dataset_path = os.path.join(args.dataset_path, args.dataset_type)  
    assert os.path.exists(dataset_path)  # 确定该数据集是否存在 

    # 每一个演示轨迹存在 "data" 数据中，命名格式为 "demo_#", 其中#为数字
    f = h5py.File(dataset_path, "r")  # 打开演示数据集 open file
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos]) # 将演示列表按剧集递增顺序排列
    demos = [demos[i] for i in inds]


    #---- 测试 robosuite demonstrations 数据 ----#
    test_robosuite_demonstrations(f, demos)


    #---- 转换数据 ----#
    states = []
    actions = []
    dones = []
    rewards = []
    next_states = []
    lengths = []

    STATES = np.array([])
    init = True
    for ep in demos:
        s = f["data/{}/states".format(ep)][()]
        if init:
            STATES = s
            init =False
        np.vstack((STATES, s))
    scale = scale_select(STATES)

    for ep in demos:
        state = f["data/{}/states".format(ep)][()]
        action = f["data/{}/actions".format(ep)][()]
        next_state = f["data/{}/next_states".format(ep)][()]
        reward = f["data/{}/rewards".format(ep)][()]
        done = f["data/{}/dones".format(ep)][()]
        horizon = f["data/{}/horizon".format(ep)][()]

        # state = scale * state
        # next_state = scale * next_state

        if args.obs_dim is not None:
            state = select_state(state)
            next_state = select_state(next_state)
            print("state.shape==========",state.shape)

        if args.failed > 0:
            print("state.shape==========",state.shape)
            state = failed_state(args.failed, state)
            action = failed_state(args.failed, action)
            next_state = failed_state(args.failed, next_state)
            reward = failed_state(args.failed, reward)
            done = failed_state(args.failed, done)
            print("长度:", horizon)
            horizon -= args.failed
            print("更改为失败轨迹:", horizon)
            print("failed.shape==========",state.shape)

        states.append(tuple(state))
        actions.append(tuple(action))
        next_states.append(tuple(next_state))
        dones.append(tuple(done.tolist()))
        rewards.append(tuple(reward.tolist()))
        lengths.append(int(horizon))


    demonstrations_trajs = defaultdict(list)
    demonstrations_trajs["states"] = states             # state:list state[0]:tuple state[0][0]:numpy.ndarray
    demonstrations_trajs["actions"] = actions           # action:list action[0]:tuple action[0][0]:numpy.ndarray
    demonstrations_trajs["dones"] = dones               # done:list done[0]:tuple done[0][0] bool
    demonstrations_trajs["rewards"] = rewards           # reward:list reward[0]:tuple reward[0][0]:float
    demonstrations_trajs["next_states"] = next_states
    demonstrations_trajs["lengths"] = lengths           # lengths:list lengths[0]:int


    #---- 保存数据 ----#
    env_args = json.loads(f["data"].attrs["env_args"])
    env_name = env_args["env_name"]
    robots = env_args["robots"][0]
    dataset_output_dir = os.path.join(args.output_dir, env_name, robots)
    if not os.path.exists(dataset_output_dir):
        os.makedirs(dataset_output_dir)


    # 保存 json格式的 env 训练参数
    del env_args["env_kwargs"]  # 移除不必要的keys
    del env_args["type"]
    iqlearn_dataset_json = os.path.join(dataset_output_dir, f'{env_name}_{robots}_json.json')
    with open(iqlearn_dataset_json,"w") as j:
        json.dump(env_args, j, indent=4)

    # 保存 pkl格式的 iqlearn 训练数据
    r = 0
    for i in range(len(rewards)):
        r += np.sum(rewards[i])/len(rewards)
    iqlearn_dataset_pkl = os.path.join(dataset_output_dir, f'{len(rewards)}_demo_{round(r,2)}.pkl')
    with open(iqlearn_dataset_pkl, 'wb') as p:
        pickle.dump(demonstrations_trajs, p)
    iqlearn_dataset_txt = os.path.join(dataset_output_dir, f'{len(rewards)}_demo_{round(r,2)}_scale.pkl')
    with open(iqlearn_dataset_txt, 'wb') as p:
        pickle.dump(scale, p)


    #---- 测试数据 iqlearn demonstrations 数据 ----#
    test_iqlearn_demonstrations(iqlearn_dataset_pkl, dataset_output_dir)
    print("scale:", scale)


if __name__ == '__main__':
    #---- 输入参数 ----#
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='dataset path containing demos to be mixed')
    parser.add_argument('--dataset_type', type=str, default='low_dim.hdf5', help='dataset type')
    parser.add_argument('--output_dir', type=str, required=True, help='output dir contain the mixed demonstration')
    parser.add_argument('--obs_dim', type=str, required=False, default=None, help='output dir contain the mixed demonstration')
    parser.add_argument('--failed', type=int, required=False, default=0, help='output dir contain the mixed demonstration')
    args = parser.parse_args()


    #---- 转换演示数据 ----#
    mix_demonstration(args)


