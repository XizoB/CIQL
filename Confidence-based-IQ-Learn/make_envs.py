import gym
import robosuite as suite

from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_controller_config
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor

from wrappers.atari_wrapper import ScaledFloatFrame, FrameStack, FrameStackEager, PyTorchFrame
from wrappers.normalize_action_wrapper import check_and_normalize_box_actions

import envs
import numpy as np
import os

# Register all custom envs
envs.register_custom_envs()

def make_dcm(cfg):
    import dmc2gym
    """Helper function to create dm_control environment"""
    if cfg.env.name == 'dmc_ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    elif cfg.env.name == 'dmc_point_mass_easy':
        domain_name = 'point_mass'
        task_name = 'easy'
    else:
        domain_name = cfg.env.name.split('_')[1]
        task_name = '_'.join(cfg.env.name.split('_')[2:])
    
    if cfg.env.from_pixels:
        # Set env variables for Mujoco rendering
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["EGL_DEVICE_ID"] = os.environ["CUDA_VISIBLE_DEVICES"]

        # per dreamer: https://github.com/danijar/dreamer/blob/02f0210f5991c7710826ca7881f19c64a012290c/wrappers.py#L26
        camera_id = 2 if domain_name == 'quadruped' else 0

        env = dmc2gym.make(domain_name=domain_name,
                        task_name=task_name,
                        seed=cfg.seed,
                        visualize_reward=False,
                        from_pixels=True,
                        height=cfg.env.image_size,
                        width=cfg.env.image_size,
                        frame_skip=cfg.env.action_repeat,
                        camera_id=camera_id)

        print(env.observation_space.dtype)
        # env = FrameStack(env, k=cfg.env.frame_stack)
        env = FrameStackEager(env, k=cfg.env.frame_stack)
        
    else:
        env = dmc2gym.make(domain_name=domain_name,
                        task_name=task_name,
                        seed=cfg.seed,
                        visualize_reward=True)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env

def make_atari(env):
    env = AtariWrapper(env)
    env = PyTorchFrame(env)
    env = FrameStack(env, 4)
    return env

def is_atari(env_name):
    return env_name in ['PongNoFrameskip-v4', 
                        'BreakoutNoFrameskip-v4', 
                        'SpaceInvadersNoFrameskip-v4', 
                        'BeamRiderNoFrameskip-v4',
                        'QbertNoFrameskip-v4',
                        'SeaquestNoFrameskip-v4']

def make_robosuite(args):
    # 机械臂控制器参数
    controller_config = load_controller_config(default_controller=args.env.controller_configs)

    # env 参数配置
    config = {
        "env_name": args.env.env_name,                              # 任务
        "robots": args.env.robots,                                  # 机器人型号
        "env_configuration": args.env.env_configuration,
        "controller_configs": controller_config,                    # 机械臂控制器参数
        "gripper_types": args.env.gripper_types,                    # 夹爪类型
        "initialization_noise": args.env.initialization_noise,
        "use_object_obs": args.env.use_object_obs,                  # 目标物信息
        "use_camera_obs": args.env.use_camera_obs,                  # 相机信息 需要 has_offscreen_renderer
        "reward_scale": args.env.reward_scale,
        "reward_shaping": args.env.reward_shaping,                  # 使用奖励工程, 细化奖励密度
        # "placement_initializer": None,                              # 设置目标物的随机初始位置
        "has_renderer": args.env.has_renderer,                      # 训练可视化
        "has_offscreen_renderer": args.env.has_offscreen_renderer,  # 用于 Image 训练与 has_renderer 二选一
        "render_camera": args.env.render_camera,                    # 训练可视化视角 默认 "frontview" 
        "render_collision_mesh": args.env.render_collision_mesh,
        "render_visual_mesh": args.env.render_visual_mesh,
        "render_gpu_device_id": args.env.render_gpu_device_id,
        "control_freq": args.env.control_freq,                      # 20hz 输出控制指令
        "horizon": args.env.horizon,                                # 每一个轨迹的长度 Horizon
        "ignore_done": args.env.ignore_done,                        # 用于执行没有一个明确完成目标的任务，如 Ant Humanoid
        "hard_reset": args.env.hard_reset,                          # 重新导入模型和启动仿真
        "camera_names": args.env.camera_names,                      # 相机类型 默认 "agentview"
        "camera_heights": args.env.camera_heights,                  # image height
        "camera_widths": args.env.camera_widths,                    # image width
        "camera_depths": args.env.camera_depths,                    # 
    }

    # 创建 env
    env = suite.make(**config)

    # 使用Gym环境包装 env
    env = GymWrapper(env)
    
    return env

def make_env(args, monitor=True):
    # dmc 控制
    if 'dmc' in args.env.name:
        env = make_dcm(args)
    # Robosuite 控制
    elif 'robosuite' in args.env.name:
        env = make_robosuite(args)
    # gym 控制
    else:
        env = gym.make(args.env.name)
    
    if monitor:
        env = Monitor(env, "gym")

    # atari 控制
    if is_atari(args.env.name):
        env = make_atari(env)

    # 对 env 动作归一化
    env = check_and_normalize_box_actions(env)  #  Normalize box actions to [-1, 1]
    return env
