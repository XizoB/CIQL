"""
A script to collect a batch of human demonstrations that can be used
to generate a learning curriculum (see `demo_learning_curriculum.py`).

The demonstrations can be played back using the `playback_demonstrations_from_pkl.py`
script.
"""

import os
import shutil
import time
import argparse
import datetime
import h5py
from glob import glob
import numpy as np
import json

import robosuite as suite
from robosuite import load_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper, GymWrapper
from robosuite.utils.input_utils import input2action



def copy_demonstrations_to_demo(f, demo_ep, ep_data_grp):

    state = f["data/{}/states".format(demo_ep)][()]
    action = f["data/{}/actions".format(demo_ep)][()]
    next_state = f["data/{}/next_states".format(demo_ep)][()]
    reward = f["data/{}/rewards".format(demo_ep)][()]
    done = f["data/{}/dones".format(demo_ep)][()]
    horizon = f["data/{}/horizon".format(demo_ep)][()]

    print(next_state.shape)

    ep_data_grp.create_dataset("states", data=state)
    ep_data_grp.create_dataset("actions", data=action)
    ep_data_grp.create_dataset("next_states", data=next_state)
    ep_data_grp.create_dataset("rewards", data=reward)
    ep_data_grp.create_dataset("dones", data=done)
    ep_data_grp.create_dataset("horizon", data=horizon)
    ep_data_grp.attrs["model_file"] = f["data/{}".format(demo_ep)].attrs["model_file"]


def gather_demonstrations_as_hdf5(args):

    # ------ 所有演示数据所在 merge_directory 文件夹
    merge_directory = args.merge_directory
    raw_directory = os.path.join(merge_directory, "raw")
    print("merge_directory:", merge_directory)

    # ------ 在 merge_directory 中构建 demo.hdf5 与 robosuite_demo.hdf5
    demo_hdf5_path = os.path.join(merge_directory, "demo.hdf5")
    robosuite_demo_hdf5_path = os.path.join(merge_directory, "robosuite_demo.hdf5")

    f_demo = h5py.File(demo_hdf5_path, "w")
    f_robosuite_demo = h5py.File(robosuite_demo_hdf5_path, "w")

    # ------ 在一个 data 组的属性中存储数据
    grp_demo = f_demo.create_group("data")  
    grp_robosuite_demo = f_robosuite_demo.create_group("data")


    demo_num_eps = 0
    robosuite_demo_num_eps = 0

    for ep_directory in os.listdir(raw_directory):
        print("ep_directory:", ep_directory)

        # ------ 打开每个子文件夹中的 demo.hdf5 与 robosuite_demo.hdf5
        demo_paths = os.path.join(raw_directory, ep_directory, "demo.hdf5")
        robosuite_demo_paths = os.path.join(raw_directory, ep_directory, "robosuite_demo.hdf5")

        f_ep_demo = h5py.File(demo_paths, "r")  # 打开演示数据集 open file
        f_ep_robosuite_demo = h5py.File(robosuite_demo_paths, "r")  # 打开演示数据集 open file

        # ------ 在 merge_directory 中的 demo 中添加
        demo_keys = list(f_ep_demo["data"].keys())
        for ep in demo_keys:
            demo_num_eps += 1
            print("add_demo_{}".format(demo_num_eps))

            ep_data_grp = grp_demo.create_group("demo_{}".format(demo_num_eps))
            copy_demonstrations_to_demo(f_ep_demo, ep, ep_data_grp)

    
        # ------ 在 merge_directory 中的 robosuite_demo 中添加
        robosuite_demo_keys = list(f_ep_robosuite_demo["data"].keys())
        for robosuite_demo_ep in robosuite_demo_keys:
            robosuite_demo_num_eps += 1
            print("add_robosuite_demo_{}".format(robosuite_demo_num_eps))

            ep_robosuite_data_grp = grp_robosuite_demo.create_group("demo_{}".format(robosuite_demo_num_eps))
            copy_demonstrations_to_demo(f_ep_robosuite_demo, robosuite_demo_ep, ep_robosuite_data_grp)


    # ------ 写入数据集属性
    now = datetime.datetime.now()
    grp_demo.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp_demo.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp_demo.attrs["repository_version"] = suite.__version__
    grp_demo.attrs["env"] = f_ep_demo["data"].attrs["env"]
    grp_demo.attrs["env_args"] = f_ep_demo["data"].attrs["env_args"]

    grp_robosuite_demo.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp_robosuite_demo.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp_robosuite_demo.attrs["repository_version"] = suite.__version__
    grp_robosuite_demo.attrs["env"] = f_ep_robosuite_demo["data"].attrs["env"]
    grp_robosuite_demo.attrs["env_args"] = f_ep_robosuite_demo["data"].attrs["env_args"]

    print("总共demo数量:", len(grp_demo.keys()))

    f_demo.close()
    f_robosuite_demo.close()


if __name__ == "__main__":
    #---- 输入参数 ---#
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge_directory", type=str)
    args = parser.parse_args()

    gather_demonstrations_as_hdf5(args)  # 把记录的数据转换为演示重现的数据






# demo_keys = list(ep_f_demo["data"].keys())
# for demo_ep in demo_keys:
#     num_eps += 1

#     ep_data_grp = grp_demo.create_group("demo_{}".format(num_eps))

#     state = ep_f_demo["data/{}/states".format(demo_ep)][()]
#     action = ep_f_demo["data/{}/actions".format(demo_ep)][()]
#     next_state = ep_f_demo["data/{}/next_states".format(demo_ep)][()]
#     reward = ep_f_demo["data/{}/rewards".format(demo_ep)][()]
#     done = ep_f_demo["data/{}/dones".format(demo_ep)][()]
#     horizon = ep_f_demo["data/{}/horizon".format(demo_ep)][()]

#     ep_data_grp.create_dataset("states", data=state)
#     ep_data_grp.create_dataset("actions", data=action)
#     ep_data_grp.create_dataset("next_states", data=next_state)
#     ep_data_grp.create_dataset("rewards", data=reward)
#     ep_data_grp.create_dataset("dones", data=done)
#     ep_data_grp.create_dataset("horizon", data=horizon)
#     ep_data_grp.attrs["model_file"] = ep_f_demo["data/{}".format(demo_ep)].attrs["model_file"]

#     print("add_demo_{}".format(num_eps))
