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


def collect_human_trajectory(env, device, arm, env_configuration):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """

    env.reset()

    # ID = 2 always corresponds to agentview
    env.render()

    is_first = True

    task_completion_hold_count = -1  # 达到目标后计数10个时间步
    device.start_control()

    # 循环，直到我们从输入获得重置或任务完成
    while True:
        # 设置机器人
        active_robot = env.robots[0] if env_configuration == "bimanual" else env.robots[arm == "left"]

        # 获得新的动作指令
        action, grasp = input2action(
            device=device,
            robot=active_robot,
            active_arm=arm,
            env_configuration=env_configuration
        )

        # 仿真
        env.step(action)
        env.render()

        # 如果没有动作指令, 重置任务
        if action is None:
            break

        # 如果我们完成了任务也会中断
        if task_completion_hold_count == 0:
            break

        # 状态检测，以检查连续10个时间步是否成功
        if env._check_success():
            if task_completion_hold_count > 0:
                task_completion_hold_count -= 1  # latched state, decrement count
            else:
                task_completion_hold_count = 10  # reset count on first success timestep
        else:
            task_completion_hold_count = -1  # null the counter if there's no success

    # 数据收集集结束时的清理
    env.close()


def gather_demonstrations_as_hdf5(directory, out_dir, env_args, is_robosuite):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file. 
        env_args (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    # 创建存储数据的文件
    if is_robosuite:
        hdf5_path = os.path.join(out_dir, "robosuite_demo.hdf5")
    else:
        hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")
    grp = f.create_group("data")  # 在一个 "data" 组的属性中存储数据

    num_eps = 0
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        
        if is_robosuite:
            state_paths = os.path.join(directory, ep_directory, "robosuite_state_*.npz")
        else:
            state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []
        rewards = []
        dones = []

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
            for ai in dic["rewards"]:
                rewards.append(ai)
            for ai in dic["dones"]:
                dones.append(ai)

        # 跳过演示采集最后的 按下Esc 记录的数据, 如果遇到状态为空直接跳过，不构建demo.hdf5数据文件
        if len(states) == 0:
            continue

        # 删除最后一个状态, 这是因为当数据收集器包装器记录状态和动作
        # 先记录初始状态                  state += 1 
        # 输入动作仿真, 记录 action += 1  state += 1 (下一时刻状态)
        # 最后得到的数据是   action = n   state = n+1
        next_states = states[1:]
        del states[-1]
        assert (len(states) == len(actions) 
                and len(states) == len(rewards) 
                and len(states) == len(dones)
                and np.allclose(next_states[:-1],states[1:])
                )

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # 将模型XML存储为属性
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # 写入状态和动作数据集
        print("")
        print("demo_{}_horizon:{}".format(num_eps, len(states)))
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))
        ep_data_grp.create_dataset("next_states", data=np.array(next_states))
        ep_data_grp.create_dataset("rewards", data=rewards)
        ep_data_grp.create_dataset("dones", data=dones)
        ep_data_grp.create_dataset("horizon", data=len(states))

    # 写入数据集属性
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_args"] = env_args

    f.close()


if __name__ == "__main__":
    #---- 输入参数 ---#
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(suite.models.assets_root, "demonstrations"),
    )
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument("--robots", nargs="+", type=str, default="Panda", help="Which robot(s) to use in the env")
    parser.add_argument("--config", type=str, default="single-arm-opposed",
                        help="Specified environment configuration if necessary")
    parser.add_argument("--arm", type=str, default="right", help="Which arm to control (eg bimanual) 'right' or 'left'")
    parser.add_argument("--camera", type=str, default="agentview", help="Which camera to use for collecting demos")
    parser.add_argument("--controller", type=str, default="OSC_POSE",
                        help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'")
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="How much to scale position user inputs")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="How much to scale rotation user inputs")
    args = parser.parse_args()


    #---- 创建 env ----#
    # 机械臂控制器参数
    controller_config = load_controller_config(default_controller=args.controller)

    # env 参数配置
    config = {
        "env_name": args.environment,               # 任务
        "robots": args.robots,                      # 机器人型号
        "env_configuration": "default",
        "controller_configs": controller_config,    # 机械臂控制器参数
        "gripper_types": "default",                 # 夹爪类型
        "initialization_noise": "default",
        "table_full_size": (0.8, 0.8, 0.05),
        "table_friction": (1., 5e-3, 1e-4),
        "use_object_obs": True,                     # 目标物信息
        "use_camera_obs": False,                    # 相机信息 需要 has_offscreen_renderer
        "reward_scale": 1.0,
        "reward_shaping": True,                     # 使用奖励工程, 细化奖励密度
        "placement_initializer": None,
        "has_renderer": True,                       # 训练可视化
        "has_offscreen_renderer": False,            # 用于 Image 训练与 has_renderer 二选一
        "render_camera": "robot0_robotview",               # 训练可视化视角 默认 "frontview" 
        "render_collision_mesh": False,
        "render_visual_mesh": True,
        "render_gpu_device_id": -1,
        "control_freq": 20,                         # 20hz 输出控制指令
        "horizon": 100,                             # 每一个轨迹的长度 Horizon
        "ignore_done": True,                       # 
        "hard_reset": False,
        "camera_names": "agentview",                # 相机类型 默认 "agentview"
        "camera_heights": 84,                       # image height
        "camera_widths": 84,                        # image width
        "camera_depths": False,                     # 
    }

    # 检查是否使用多臂环境，如果是，使用env_configuration参数
    if "TwoArm" in args.environment:
        config["env_configuration"] = args.config

    # 创建 env
    env = suite.make(**config)

    #---- 包装 env ----#
    # 使用Gym环境包装 env
    env = GymWrapper(env)

    # 用可视化包装器包装 env
    env = VisualizationWrapper(env)

    # 用数据收集包装器包装 env
    t1, t2 = str(time.time()).split(".")
    output_dir = os.path.join(args.directory, args.environment, "Raw", "{}_{}_{}".format(args.robots[0], t1, t2))
    tmp_directory = os.path.join(output_dir, "tmp")
    env = DataCollectionWrapper(env, tmp_directory)


    #---- 初始化遥操作设备 ----#
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        env.viewer.add_keypress_callback("any", device.on_press)
        env.viewer.add_keyup_callback("any", device.on_release)
        env.viewer.add_keyrepeat_callback("any", device.on_press)
    elif args.device == "omega":
        from robosuite.devices import Omega

        device = Omega(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        env.viewer.add_keypress_callback("any", device.on_press)
        env.viewer.add_keyup_callback("any", device.on_release)
        env.viewer.add_keyrepeat_callback("any", device.on_press)
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    else:
        raise Exception(
            "Invalid device choice: choose either 'keyboard' or 'spacemouse'."
        )


    #---- 收集演示数据 ----#
    env_kwargs = {}
    for key in config:
        if key == "env_name":
            continue
        if key == "controller_configs":
            env_kwargs[key] = config[key]
            env_kwargs[key]["damping"] = config[key]["damping_ratio"]
            env_kwargs[key]["damping_limits"] = config[key]["damping_ratio_limits"]
            del env_kwargs[key]["damping_ratio"]
            del env_kwargs[key]["damping_ratio_limits"]
        env_kwargs[key]= config[key]
    config["env_kwargs"] = env_kwargs
    config["type"] = 1
    env_args = json.dumps(config, indent=4)  # 获取对控制器配置的引用并将其转换为json编码的字符串

    while True:
        collect_human_trajectory(env, device, args.arm, args.config)
        gather_demonstrations_as_hdf5(tmp_directory, output_dir, env_args, is_robosuite=False)  # 把记录的数据转换为演示重现的数据
        gather_demonstrations_as_hdf5(tmp_directory, output_dir, env_args, is_robosuite=True)   # 把记录的数据转换为robosuite训练的数据
