from collections import deque
from re import L
import numpy as np
import random
import torch

from torch.autograd import Variable
from wrappers.atari_wrapper import LazyFrames
from dataset.expert_dataset import ExpertDataset


class Memory(object):
    def __init__(self, memory_size: int, device, seed: int = 0) -> None:
        random.seed(seed)
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)
        self.device = device

    def add(self, experience) -> None:
        # 把每一个元组转换为tensor
        (state, next_state, action, reward, done_no_lim) = experience
        state = torch.as_tensor(state, dtype=torch.float)
        next_state = torch.as_tensor(next_state, dtype=torch.float)
        action = torch.as_tensor(action, dtype=torch.float)
        reward = torch.as_tensor(reward, dtype=torch.float)
        done_no_lim = torch.as_tensor(done_no_lim, dtype=torch.float)
        experience = (state, next_state, action, reward, done_no_lim)
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

    def save(self, path):
        b = np.asarray(self.buffer)
        print(b.shape)
        np.save(path, b)

    def load(self, path, choice_trajs, subopt_class_num, num_trajs, sample_freq, seed, label_ratio, sparse_sample):
        # If path has no extension add npy
        if not path.endswith("pkl"):
            path += '.npy'
        data = ExpertDataset(path, self.device, choice_trajs, subopt_class_num, num_trajs, sample_freq, seed, label_ratio, sparse_sample) # 实例化 ExpertDataset
        self.sample_traj = data.sample_traj() # 在导入专家文件的时候采样出本次实验需要用到的专家轨迹 给def sample_expert_traj 用的
        # data = np.load(path, allow_pickle=True)
        for i in range(len(data)):
            self.add(data[i])  # 向replaybuffer中添加一个data中MDP元组

    def sample(self, batch_size: int, continuous: bool = True):
        """
        直接在 Buffer 中采样
        """
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            indexes = [i for i in range(rand, rand + batch_size)]
            return [self.buffer[i] for i in indexes], indexes
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes], indexes

    def get_samples(self, batch_size, continuous=False):
        batch, indexes = self.sample(batch_size, continuous)
        batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

        # Scale obs for atari. TODO: Use flags
        if isinstance(batch_state[0], LazyFrames):
            # Use lazyframes for improved memory storage (same as original DQN)
            batch_state = np.array(batch_state) / 255.0
        if isinstance(batch_next_state[0], LazyFrames):
            batch_next_state = np.array(batch_next_state) / 255.0
        
        batch_state = torch.stack(batch_state).to(self.device)
        batch_next_state = torch.stack(batch_next_state).to(self.device)
        batch_action = torch.stack(batch_action).to(self.device)
        batch_reward = torch.stack(batch_reward).to(self.device).unsqueeze(1)
        batch_done = torch.stack(batch_done).to(self.device).unsqueeze(1)
        batch_transition = batch_state, batch_next_state, batch_action, batch_reward, batch_done
        return batch_transition, indexes

    def exp_sample(self, batch_size: int, continuous: bool = True):
        """
        直接在 Buffer 中采样专家样本与对 self.conf 进行采样
        """
        # 截断专家样本的置信度，使得其在(0,2)的区间内
        all_conf = Variable(self.conf)
        all_conf_mean = Variable(all_conf.mean())
        confidence = all_conf / all_conf_mean
        confidence.clamp_(0, 2)
        with torch.no_grad():
            self.conf = confidence
        self.conf.requires_grad = True

        # 随机采样指定 batch 大小的样本
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            self.indexes = [i for i in range(rand, rand + batch_size)]
            return [self.buffer[i] for i in self.indexes], self.conf[self.indexes]
        else:
            self.indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in self.indexes], self.conf[self.indexes]

    def get_exp_samples(self, batch_size, conf, device):
        """
        采样指定 Batch 大小的专家样本包括MDP五元组以及batch_conf和全局self.conf
        batch_state, batch_next_state, batch_action, batch_reward, batch_done, batch_conf, self.conf
        """
        self.conf = conf
        batch, batch_conf = self.exp_sample(batch_size, False)

        batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

        # Scale obs for atari. TODO: Use flags
        if isinstance(batch_state[0], LazyFrames):
            # Use lazyframes for improved memory storage (same as original DQN)
            batch_state = np.array(batch_state) / 255.0
        if isinstance(batch_next_state[0], LazyFrames):
            batch_next_state = np.array(batch_next_state) / 255.0

        batch_state = torch.stack(batch_state).to(device)
        batch_next_state = torch.stack(batch_next_state).to(device)
        batch_action = torch.stack(batch_action).to(device)
        batch_reward = torch.stack(batch_reward).to(device).unsqueeze(1)
        batch_done = torch.stack(batch_done).to(device).unsqueeze(1)

        return batch_state, batch_next_state, batch_action, batch_reward, batch_done, batch_conf, self.conf

    def sample_expert_traj(self):
        """
        专家样本
        """
        label_states_traj, _, label_actions_traj = self.sample_traj
        sample_states_traj = []
        sample_actions_traj = []

        for _ in range(2):
            k = random.randint(0, len(label_states_traj)-1)
            sample_states_traj.append(torch.as_tensor(np.vstack(label_states_traj[k]), dtype=torch.float, device=self.device))
            sample_actions_traj.append(torch.as_tensor(np.vstack(label_actions_traj[k]), dtype=torch.float, device=self.device))

        return sample_states_traj, sample_actions_traj

    def sample_all_expert_traj(self):
        """
        专家样本
        """
        return self.sample_traj