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
import random
import numpy as np
import json

import robosuite
from robosuite.wrappers import GymWrapper

from robosuite.utils.mjcf_utils import postprocess_model_xml

def main(args):
    #---- 导入参数 ---#
    demo_path = args.folder
    hdf5_path = os.path.join(demo_path, "demo.hdf5")
    f = h5py.File(hdf5_path, "r")
    config = json.loads(f["data"].attrs["env_args"])
    del config["env_kwargs"]  # 移除不必要的keys
    del config["type"]
    

    #---- 创建 env ---#
    env = robosuite.make(**config)
    env = GymWrapper(env)


    demos = list(f["data"].keys())  # 所有演示集的列表
    print("demos:", demos)

    for ep in demos:
        print("Playing back random episode... (press ESC to quit)")

        # #  随机选择一个演示集
        # ep = random.choice(demos)
        print("ep:", ep)

        # 使用演示数据集 data 属性中读取模型xml
        model_xml = f["data/{}".format(ep)].attrs["model_file"]

        env.reset()
        xml = postprocess_model_xml(model_xml)
        env.reset_from_xml_string(xml)
        env.sim.reset()
        env.viewer.set_camera(0)

        # load the flattened mujoco states
        states = f["data/{}/states".format(ep)][()]

        if args.use_actions  == "True":

            # load the initial state
            env.sim.set_state_from_flattened(states[0])
            env.sim.forward()

            # load the actions and play them back open-loop
            actions = np.array(f["data/{}/actions".format(ep)][()])
            num_actions = actions.shape[0]


            for j, action in enumerate(actions):
                next_state, reward, done, info = env.step(action)
                env.render()

                if j < num_actions - 1:
                    # 判断仿真的状态是否与记录的状态一样 recorded states
                    state_playback = env.sim.get_state().flatten()
                    if not np.all(np.equal(states[j + 1], state_playback)):
                        err = np.linalg.norm(states[j + 1] - state_playback)
                        print(f"[warning] playback diverged by {err:.5f} for ep {ep} at step {j}")
            
            print("Horizons: {}\tUse actions: {}\t".format(num_actions, args.use_actions))

        else:

            # force the sequence of internal mujoco states one by one
            j = 0
            for state in states:
                env.sim.set_state_from_flattened(state)
                env.sim.forward()
                env.render()

                if j < len(states):
                    # 判断仿真的状态是否与记录的状态一样 recorded states
                    state_playback = env.sim.get_state().flatten()
                    if not np.all(np.equal(states[j], state_playback)):
                        err = np.linalg.norm(states[j] - state_playback)
                        print(f"[warning] playback diverged by {err:.5f} for ep {ep} at step {j}")
                j += 1
            print("Horizons: {}\tUse states: {}\t".format(len(states), args.use_actions))


    f.close()


if __name__ == "__main__":
    #---- 输入参数 ---#
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to your demonstration folder that contains the demo.hdf5 file, e.g.: "
             "'path_to_assets_dir/demonstrations/YOUR_DEMONSTRATION'"
    ),
    parser.add_argument(
        "--use-actions", 
        default=True,
    )


    #---- 回放演示数据 ---#
    args = parser.parse_args()
    main(args)