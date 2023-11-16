"""
A script to visualize dataset trajectories by loading the simulation states
one by one or loading the first state and playing actions back open-loop.
The script can generate videos as well, by rendering simulation frames
during playback. The videos can also be generated using the image observations
in the dataset (this is useful for real-robot datasets) by using the
--use-obs argument.

Args:
    dataset (str): path to hdf5 dataset

    filter_key (str): if provided, use the subset of trajectories
        in the file that correspond to this filter key

    n (int): if provided, stop after n trajectories are processed

    use-obs (bool): if flag is provided, visualize trajectories with dataset 
        image observations instead of simulator

    use-actions (bool): if flag is provided, use open-loop action playback 
        instead of loading sim states

    render (bool): if flag is provided, use on-screen rendering during playback
    
    video_path (str): if provided, render trajectories to this video file path

    video_skip (int): render frames to a video every @video_skip steps

    render_image_names (str or [str]): camera name(s) / image observation(s) to 
        use for rendering on-screen or to video

    first (bool): if flag is provided, use first frame of each episode for playback
        instead of the entire episode. Useful for visualizing task initializations.

Example usage below:

    # force simulation states one by one, and render agentview and wrist view cameras to video
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --render_image_names agentview robot0_eye_in_hand \
        --video_path /tmp/playback_dataset.mp4

    # playback the actions in the dataset, and render agentview camera during playback to video
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-actions --render_image_names agentview \
        --video_path /tmp/playback_dataset_with_actions.mp4

    # use the observations stored in the dataset to render videos of the dataset trajectories
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --use-obs --render_image_names agentview_image \
        --video_path /tmp/obs_trajectory.mp4

    # visualize initial states in the demonstration data
    python playback_dataset.py --dataset /path/to/dataset.hdf5 \
        --first --render_image_names agentview \
        --video_path /tmp/dataset_task_inits.mp4
"""

import hydra
import argparse
import imageio
import gym
import numpy as np
import torch

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils

from omegaconf import DictConfig, OmegaConf
from robomimic.envs.env_base import EnvBase, EnvType
from itertools import count


from omegaconf import DictConfig, OmegaConf

from make_envs import make_env
from agent import make_agent


# Define default cameras to use for each env type
DEFAULT_CAMERAS = {
    EnvType.ROBOSUITE_TYPE: ["agentview"],
    EnvType.IG_MOMART_TYPE: ["rgb"],
    EnvType.GYM_TYPE: ValueError("No camera names supported for gym type env!"),
}


def playback_trajectory_with_env(
    env,
    agent,
    HORIZON,
    render=False, 
    video_writer=None, 
    video_skip=5, 
    camera_names=None,
    first=False,
):
    """
    Helper function to playback a single trajectory using the simulator environment.
    If @actions are not None, it will play them open-loop after loading the initial state. 
    Otherwise, @states are loaded one by one.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load
        actions (np.array): if provided, play actions back open-loop instead of using @states
        render (bool): if True, render on-screen
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    assert isinstance(env, EnvBase)

    write_video = (video_writer is not None)
    video_count = 0
    assert not (render and write_video)

    # load the initial state
    state = env.reset()
    rewards = []
    dones = []
    for step in range(HORIZON):
        
        
        env.render()
        action = agent.choose_action(state, sample=False)
        next_state, reward, done, info = env.step(action)
        rewards.append(reward)
        dones.append(done)
            


        # on-screen render
        if render:
            env.render(mode="human", camera_name=camera_names[0])

        # video render
        if write_video:
            if video_count % video_skip == 0:
                video_img = []
                for cam_name in camera_names:
                    video_img.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam_name))
                video_img = np.concatenate(video_img, axis=1) # concatenate horizontally
                video_writer.append_data(video_img)
            video_count += 1

        if done:
            break
        state = next_state

        if first:
            break


def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(config_path="conf/rl", config_name="config")
def main(cfg: DictConfig, parse_args):
    args = get_args(cfg)

    HORIZON = int(parse_args.env.horizon)

    env = make_env(parse_args, monitor=False)
    env.seed(parse_args.seed + 100)
    agent = make_agent(env, parse_args)

    policy_file = f'{parse_args.eval.policy}'
    print(f'Loading policy from: {policy_file}', f'_{parse_args.env.name}')
    print("")

    agent.load(hydra.utils.to_absolute_path(policy_file), f'_{parse_args.env.name}')

    ###### ------------------------------------------------- ######
    # some arg checking
    write_video = (args.video_path is not None)
    assert not (args.render and write_video) # either on-screen or video but not both

    if args.render:
        # on-screen rendering can only support one camera
        assert len(args.render_image_names) == 1

    # maybe dump video
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)


    # supply actions if using open-loop action playback
    playback_trajectory_with_env(
        env=env,
        agent=agent,
        HORIZON=HORIZON,
        render=args.render, 
        video_writer=video_writer, 
        video_skip=args.video_skip,
        camera_names=args.render_image_names,
        first=args.first,
    )

    if write_video:
        video_writer.close()




if __name__ == "__main__":
#---- 输入参数 ---#
    parser = argparse.ArgumentParser()
    # 可视化
    parser.add_argument(
        "--render",
        action='store_true',
        help="on-screen rendering",
    )

    # 将数据集回放的视频转储到指定路径
    parser.add_argument(
        "--video_path",
        type=str,
        default="/root/RoboLearn/Test/Test_launch/Data_train/playback_demonstration/IIWA_StackObstacle.mp4",
        help="(optional) render trajectories to this video file path",
    )

    # 回放期间写入视频帧的频率
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # 要渲染的摄像机名称，或用于写入视频的图像观察结果
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs='+',
        default="frontview" "robot0_eye_in_hand" "robot0_robotview",
        help="(optional) camera name(s) / image observation(s) to use for rendering on-screen or to video. Default is"
             "None, which corresponds to a predefined camera for each env type",
    )

    # 仅使用每个演示数据的第一帧，即初始化状态
    parser.add_argument(
        "--first",
        action='store_true',
        help="use first frame of each episode",
    )

    #---- 回放演示数据 ---#
    args = parser.parse_args()
    main(parse_args=args)

