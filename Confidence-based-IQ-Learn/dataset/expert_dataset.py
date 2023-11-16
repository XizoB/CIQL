from random import sample
from typing import Any, Dict, IO, List, Tuple

import numpy as np
import pickle
import torch
import random
from torch.utils.data import Dataset
import os


class ExpertDataset(Dataset):
    """
    专家轨迹的数据集
    假设专家数据集是一个带有键 {states, actions,rewards,lengths} 的字典，其值包含一个列表
    下面给定形状的专家属性。 每个轨迹可以具有不同的长度

    专家奖励不是必需的，但可用于评估

    Assumes expert dataset is a dict with keys {states, actions, rewards, lengths} with values containing a list of
    expert attributes of given shapes below. Each trajectory can be of different length.

    Expert rewards are not required but can be useful for evaluation.

        shapes:
            expert["states"]  =  [num_experts, traj_length, state_space]
            expert["actions"] =  [num_experts, traj_length, action_space]
            expert["rewards"] =  [num_experts, traj_length]
            expert["lengths"] =  [num_experts]
    """

    def __init__(self,
                 expert_location: str,
                 device,
                 choice_trajs: str,
                 subopt_class_num: int,
                 num_trajectories: int = 4,
                 subsample_frequency: int = 20,
                 seed: int = 0,
                 label_ratio: float = 0.0,
                 sparse_sample: bool = True
                 ):
        """Subsamples an expert dataset from saved expert trajectories.

        Args:
            专家位置: 保存的专家轨迹的位置
            um_trajectories: 要采样的专家轨迹数（随机）
            subsample_frequency: 以指定的步长频率对每个轨迹进行二次采样
            确定性: 如果为真，则对确定性专家轨迹进行采样
            expert_location:          Location of saved expert trajectories.
            num_trajectories:         Number of expert trajectories to sample (randomized).
            subsample_frequency:      Subsamples each trajectory at specified frequency of steps.
            deterministic:            If true, sample determinstic expert trajectories.
        """
        # 导入专家样本
        all_trajectories = load_trajectories(expert_location, choice_trajs, subopt_class_num, num_trajectories, seed)
        self.trajectories = {}

        # Randomize start index of each trajectory for subsampling
        # start_idx = torch.randint(0, subsample_frequency, size=(num_trajectories,)).long()

        # 每个 `subsample_frequency` 步骤的子样本专家轨迹, 即从每一整条轨迹中采样片段
        # Subsample expert trajectories with every `subsample_frequency` step.
        for k, v in all_trajectories.items():
            data = v

            if k != "lengths":
                samples = []
                for i in range(num_trajectories):
                    samples.append(data[i][0::subsample_frequency]) # 相间隔 subsample_frequency 进行采样
                self.trajectories[k] = samples
            else:
                # Adjust the length of trajectory after subsampling
                self.trajectories[k] = np.array(data) // subsample_frequency

        self.i2traj_idx = {}
        self.length = self.trajectories["lengths"].sum().item()

        del all_trajectories  # Not needed anymore 这个存着专家样本中所有的元素，非常占空间资源
        traj_idx = 0
        i = 0

        # 将trajectories中“lengths”轨迹，按照每条轨迹先后顺序提取出轨迹中元素的索引元组（traj_idx,i）
        # Convert flattened index i to trajectory indx and offset within trajectory
        self.get_idx = []

        for _j in range(self.length):
            while self.trajectories["lengths"][traj_idx].item() <= i:
                i -= self.trajectories["lengths"][traj_idx].item()
                traj_idx += 1

            self.get_idx.append((traj_idx, i))
            i += 1
        

        state_dim = len(self.trajectories["states"][0][0]) # 状态维度
        action_dim = len(self.trajectories["actions"][0][0]) # 动作维度


        # self.label_states_traj = np.stack(tuple(self.trajectories["states"]))
        # self.label_next_states_traj = np.stack(tuple(self.trajectories["next_states"]))
        # self.label_actions_traj = np.stack(tuple(self.trajectories["actions"]))

        # self.label_states_traj = torch.as_tensor(self.label_states_traj, dtype=torch.float, device=device).view(-1,state_dim)
        # self.label_next_states_traj = torch.as_tensor(self.label_next_states_traj, dtype=torch.float, device=device).view(-1,state_dim)
        # self.label_actions_traj= torch.as_tensor(self.label_actions_traj, dtype=torch.float, device=device).view(-1,action_dim)
        # self.label_rewards_traj = torch.as_tensor(self.trajectories["rewards"], dtype=torch.float, device=device).sum(1).view(-1,1)
        # self.label_done_traj = torch.as_tensor(self.trajectories["dones"], dtype=torch.float, device=device).view(-1,1).squeeze(1)


        self.states_traj = []
        self.next_states_traj = []
        self.actions_traj = []
        self.rewards_traj = []
        self.length_traj = []

        for i in range(len(self.trajectories["states"])):
            self.states_traj.append(torch.as_tensor(np.vstack(self.trajectories["states"][i]), dtype=torch.float, device=device))
            self.next_states_traj.append(torch.as_tensor(np.vstack(self.trajectories["next_states"][i]), dtype=torch.float, device=device))
            self.actions_traj.append(torch.as_tensor(np.vstack(self.trajectories["actions"][i]), dtype=torch.float, device=device))
            self.rewards_traj.append(torch.as_tensor(np.vstack(self.trajectories["rewards"][i]), dtype=torch.float, device=device))
            self.length_traj.append((self.trajectories["lengths"][i]))


        # print("states_traj:", len(self.states_traj))
        # print("states_traj:", type(self.states_traj[0]))
        # print("states_traj.shape:", self.states_traj[0].shape)
        # print("actions_traj.shape:", self.actions_traj[0].shape)
        # print("rewards_traj.shape:", self.rewards_traj[0].shape)


    def __len__(self) -> int:
        """
        返回数据集的总长度
        Return the length of the dataset
        """
        return self.length

    def __getitem__(self, i):
        traj_idx, i = self.get_idx[i]

        states = self.trajectories["states"][traj_idx][i]
        next_states = self.trajectories["next_states"][traj_idx][i]

        # Rescale states and next_states to [0, 1] if are images
        if isinstance(states, np.ndarray) and states.ndim == 3:
            states = np.array(states) / 255.0
        if isinstance(states, np.ndarray) and next_states.ndim == 3:
            next_states = np.array(next_states) / 255.0

        return (states,
                next_states,
                self.trajectories["actions"][traj_idx][i],
                self.trajectories["rewards"][traj_idx][i],
                self.trajectories["dones"][traj_idx][i])
    
    def sample_traj(self):
        return self.states_traj, self.next_states_traj, self.actions_traj, self.rewards_traj, self.length_traj


def load_trajectories(expert_location: str,
                      choice_trajs: str,
                      subopt_class_num: int,
                      num_trajectories: int = 10,
                      seed: int = 0) -> Dict[str, Any]:
    """
    加载专家轨迹
    Load expert trajectories

    Args:
        专家位置：保存的专家轨迹的位置
        num_trajectories: 要采样的专家轨迹数（随机）
        确定性：如果为真，则关闭随机行为
        expert_location:          Location of saved expert trajectories.
        num_trajectories:         Number of expert trajectories to sample (randomized).
        deterministic:            If true, random behavior is switched off.

    Returns:
        包含键 {"states", "lengths"} 和可选的 {"actions", "rewards"} 和值的字典
        包含相应的专家数据属性
        Dict containing keys {"states", "lengths"} and optionally {"actions", "rewards"} with values
        containing corresponding expert data attributes.
    """
    if os.path.isfile(expert_location):
        # 从单个文件加载数据
        # Load data from single file.
        with open(expert_location, 'rb') as f:
            trajs = read_file(expert_location, f)

        rng = np.random.RandomState(seed)
        # Sample random `num_trajectories` experts.
        # 随机抽样 `num_trajectories` 专家
        perm = []
        idx = []
        if not choice_trajs:
            space_len = int(len(trajs["states"])/subopt_class_num)
            for i in range(subopt_class_num):
                perm = rng.permutation(np.arange(space_len*i,space_len*(i+1)))
                idx.append(perm[:int(num_trajectories/subopt_class_num)])
        else:
            space_len = num_trajectories
            i = subopt_class_num-1
            idx = np.arange(space_len*i,space_len*(i+1))
        idx = np.array(idx).reshape(1,-1).squeeze(0)

        for k, v in trajs.items():  # 转换为字典并从相应key中抽取idx相应的轨迹
            # if not torch.is_tensor(v):
            #     v = np.array(v)  # convert to numpy array
            trajs[k] = [v[i] for i in idx]

    else:
        raise ValueError(f"{expert_location} is not a valid path")
    return trajs


def read_file(path: str, file_handle: IO[Any]) -> Dict[str, Any]:
    """
    从输入路径读取文件 假设文件存储字典数据
    Read file from the input path. Assumes the file stores dictionary data.

    Args:
        路径: 本地或 S3 文件路径
        file_handle: 文件的文件句柄
        path:               Local or S3 file path.
        file_handle:        File handle for file.

    Returns:
        文件的字典表示
        The dictionary representation of the file.
    """
    if path.endswith("pt"):
        data = torch.load(file_handle)
    elif path.endswith("pkl"):
        data = pickle.load(file_handle)
    elif path.endswith("npy"):
        data = np.load(file_handle, allow_pickle=True)
        if data.ndim == 0:
            data = data.item()
    else:
        raise NotImplementedError
    return data
