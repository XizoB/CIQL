import torch
import pickle
import numpy as np
import argparse
import os

from collections import defaultdict

def mix_demo(folder: str, env_id: str, each_num: int):
    """
    根据文件夹中的演示创建混合演示
    Create a mixture of demonstrations based on demonstrations in the folder

    Parameters
    ----------
    folder: str
        包含要混合的演示的文件夹
        folder containing demos to be mixed
    env_id: str
        环境名称
        name of the environment
    """
    ########### 依次导入文件夹中的专家样本文件
    # random_name = []
    buffer_name = []  # 存入文件夹里面的专家样本文件
    files = os.listdir(folder)
    for file in sorted(files):
        buffer_name.append(os.path.join(folder, file))
        # random_name.append(int(file.split('_')[1][0]))


    ########### 混合专家样本数据
    epoch_return = []  # 每个专家轨迹的回报

    init_data = True
    init_action_box = True
    for i_buffer, name in enumerate(buffer_name):

        f = open(name, "rb")  # 导入专家数据
        data = pickle.load(f)

        # 'states', 'next_states', 'actions', 'rewards', 'dones', 'lengths'

        if init_data == True:
            expert_trajs = data
            init_data = False
            continue
        
        expert_trajs["states"].extend(data["states"])
        expert_trajs["next_states"].extend(data["next_states"])
        expert_trajs["actions"].extend(data["actions"])
        expert_trajs["rewards"].extend(data["rewards"])
        expert_trajs["dones"].extend(data["dones"])
        expert_trajs["lengths"].extend(data["lengths"])

        # if init_action_box:
        #     print("init_action_box:", init_action_box)
        #     init_action_box = False
        #     action_min = np.min(data["actions"],axis=0)
        #     action_max = np.max(data["actions"],axis=0)
        # else:
        #     action_min = np.vstack((action_min,np.min(data["actions"],axis=0)))
        #     action_min = np.min(action_min,axis=0)
        #     action_max = np.vstack((action_max,np.max(data["actions"],axis=0)))
        #     action_max = np.max(action_max,axis=0)

    ########### 保存Suboptimal专家数据
    with open(os.path.join(folder, f'{env_id}.pkl'), 'wb') as f:
        pickle.dump(expert_trajs, f)

    # print("动作最小: {}".format(action_min))
    # print("动作最大: {}".format(action_max))
    print("type(expert_trajs):", type(expert_trajs))
    print("expert_trajs.keys():", expert_trajs.keys())
    print("len(expert['states']):", len(expert_trajs['states']))
    print("len(info['states'][0]):", len(expert_trajs['states'][0]))
    print("len(info['next_states'][0]):", len(expert_trajs['next_states'][0][0]))
    print("epoch_return:", epoch_return)
    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # required
    parser.add_argument('--folder', type=str, required=True, help='folder containing demos to be mixed')
    parser.add_argument('--env-id', type=str, required=True, help='name of the environment')
    parser.add_argument('--each_num', type=int, default=5, required=False, help='each epsodes of num')

    args = parser.parse_args()
    mix_demo(folder=args.folder, env_id=args.env_id, each_num=args.each_num)
