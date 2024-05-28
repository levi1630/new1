''' 记忆模型
回合：
    obs_embeding embeding
    state （可能不需要）
    action
    reward
    next state
    done
    priority
    steps

    方法：
        update priority
        add
        
记忆（回合）：
    专家演示轨迹
    练习轨迹

    方法：
        编码专家轨迹 输入cfg 读取文件
        专家演示采样，用于克隆
        更新某组专家演示轨迹
        奖励函数，输入练习轨迹序号，输出一组奖励值
        -混合采样，用于训练，输出练习轨迹的序列号
        -记忆匹配，输入练习轨迹序列号匹配对应的专家轨迹序列号以及每组轨迹的匹配步数
        -奖励计算，输入两组序列号，以及对应的匹配步数，输出一组长奖励值
'''
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import re
import numpy as np
from omegaconf import ListConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
from PIL import Image
from pathlib import Path
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal
import random

def cal_similarity(pratice_embeding, demo_embeding, loss_mode='KL'):
    '''
    output (0,1)
    '''
    distance = pratice_embeding - demo_embeding

    if loss_mode == 'KL':
        # KL 散度，从 有log 的角度看 无log 分布相对 有log 分布的差异
        kl_divergence = F.kl_div(demo_embeding.softmax(dim=-1).log(), pratice_embeding.softmax(dim=-1), reduction='sum')
        return kl_divergence

class Episode(object):
    """Storage object for a single episode."""

    def __init__(self, cfg, episode_length, init_obs_embeding, init_state=None):
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.obs_embeding = torch.empty(
            (episode_length, cfg.frame_stack*cfg.latent_dim),
            dtype=torch.float32,
            device=self.device
        )
        self.obs_embeding[0] = init_obs_embeding
        self.state = torch.empty(
            (episode_length, cfg.state_dim),
            dtype=torch.float32,
            device=self.device
        )
        self.state[0] = init_state
        self.action = torch.empty(
            (cfg.episode_length, cfg.action_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.reward = torch.empty(
            (cfg.episode_length,), dtype=torch.float32, device=self.device
        )
        self.value = np.ones(episode_length, np.float32)
        self.value[0] = 0.0
        self.cumulative_reward = 0
        self.done = False
        self.truncated = False
        self._idx = 0

    @classmethod
    def from_trajectory(cls, cfg, episode_length, obs_embedings, states, action, reward, done=True):
        ###
        """Constructs an episode from a trajectory."""
        episode = cls(cfg, episode_length, obs_embedings[0], states[0])
        episode.obs_embeding[1:] = obs_embedings[1:]
        episode.state[1:] = states[1:]
        episode.action = action
        episode.reward = reward
        episode.value[1:] = episode_length-1
        episode.cumulative_reward = torch.sum(episode.reward)
        episode._idx = episode_length
        return episode

    @classmethod
    def modem_from_trajectory(cls, cfg, length, obs_embedings, states, action, reward, done=None):
        """Constructs an episode from a trajectory."""
        episode = cls(cfg, length, obs_embedings[0], states[0])
        episode.obs_embeding[1:] = obs_embedings[1:]
        episode.state[1:] = states[1:]
        episode.action = torch.tensor(
            action, dtype=episode.action.dtype, device=episode.device
        )
        episode.reward = torch.tensor(
            reward, dtype=episode.reward.dtype, device=episode.device
        )
        episode.cumulative_reward = torch.sum(episode.reward)
        episode.done = True
        episode._idx = cfg.episode_length
        return episode
    
    def __len__(self):
        return self._idx

    def __add__(self, transition):
        self.add(*transition)
        return self

    def add(self, obs_embeding, state, action, reward, done, value=1.0):
        self.obs_embeding[self._idx + 1] = obs_embeding
        self.state[self._idx + 1] = torch.tensor(
            state, dtype=self.state.dtype, device=self.state.device
        )
        self.action[self._idx] = action
        self.reward[self._idx] = reward
        self.value[self._idx] = value
        self.cumulative_reward += reward
        self.done = done
        self._idx += 1
        if self._idx == self.demo_length-1:
            self.truncated = True

    def update_value(self, practise_episode):
        assert isinstance(practise_episode, Episode), '输入应为 Episode 类型'
        with torch.no_grad():
            sim_list = [cal_similarity(self.obs_embeding[i], practise_episode.obs_embeding[i]).numpy() for i in range(self._idx)]
        learned_frame = np.array(sim_list) > self.cfg.learned_frame_threshould
        self.value[learned_frame & np.array(self.value>0.0001)] -= 0.0001
        

class Memory(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.save_device = torch.device('cpu')
        self.cal_device = torch.device("cuda")
        obs_shape = (3, *cfg.obs_shape[-2:])
        self._state_dim = cfg.state_dim
        self.demo_episodes = []
        self.practise_episodes = []
        self.idx = 0

        self._obs_encoder = self.initialize_encoder()
        # self._encode_demostrations()
        self.modem_get_demos(self.cfg)

    def initialize_encoder(self):
        from r3m import load_r3m

        r3m = load_r3m("resnet18") # resnet18, resnet34
        r3m.to(self.cal_device)
        r3m.eval()
        return r3m
    
    def obs_encoder(self, obs):
        assert len(obs)==3*self.cfg.frame_stack
        obs_embeding = []
        for i in range(self.cfg.frame_stack):
            obs_embeding.append(self._obs_encoder(torch.tensor(obs[i*3:i*3+3]).unsqueeze(0)))
        return torch.cat(obs_embeding, dim=1)

    def __len__(self):
        return self.idx

    def __add__(self, episode: Episode):
        self.add(episode)
        return self
    
    def add(self, episode: Episode):
        if episode.truncated and episode.done:
            self._replace_demo_episode(episode)  # 更换 episode 要不要刷新权重？
        self.demo_episodes[episode.demo_id].update_value(episode)
        self.practise_episodes.append(episode)  
        self.idx += 1
   
    def init_episode(self, init_obs, init_state):
        '''
        Initialize an episode from init_obs
        '''
        obs_embeding = self.obs_encoder(init_obs)
        state_embeding = torch.tensor(init_state).unsqueeze(0).to(self.cal_device)
        init_embeding = torch.cat((obs_embeding,state_embeding), dim=1)
        demo_id, similarity = self._init_obs_match_demo(init_embeding[:,0:self.cfg.latent_dim])
        episode = Episode(self.cfg, len(self.demo_episodes[demo_id]), obs_embeding, state_embeding)
        setattr(episode, 'demo_id', demo_id)
        setattr(episode, 'demo_length', len(self.demo_episodes[demo_id]))
        setattr(episode, 'demo_similarity', similarity)
        return episode, init_embeding  
    
    def add_tuple_to_episode(self, obs, state, action, reward, done, episodo: Episode):
        obs_embeding = self.obs_encoder(obs)
        episodo += (obs_embeding, state, action, reward, done)
        if len(episodo) == len(self.demo_episodes[episodo.demo_id]):
            episodo.truncated = True

    def _replace_demo_episode(self, episode: Episode):
        assert hasattr(episode, 'demo_id'), "Episode doesn't match demostration."
        setattr(episode, 'seed', self.demo_episodes[episode.demo_id].seed)
        delattr(episode, 'demo_id')
        self.demo_episodes[episode.demo_id] = episode
        print(f'Replace {episode.demo_id}-th demostration')

    def _init_obs_match_demo(self, init_embeding):
        ### 这里需要测试一下 r3m 对不同种子初始化页面的敏感程度
        demo_similarity = torch.stack([cal_similarity(init_embeding, demo_embeding.obs_embeding[0,:self.cfg.latent_dim].unsqueeze(0).to(self.cal_device))
                           for demo_embeding in self.demo_episodes])
        return torch.argmin(demo_similarity), torch.min(demo_similarity)

    def _calculate_reward_from_episode(self, episode):
        assert hasattr(episode, 'demo_id'), "Episode doesn't match demostration."
        demo = self.demo_episodes[episode.demo_id]
        
    def get_reward(self):
        ep_index = np.random.randint(0, self.idx, self.cfg.batch_size)
        return [self._calculate_reward_from_episode(self.practise_episodes[index]) for index in ep_index]

    def _encode_demostrations(self):
        data_path = self.cfg.demo_path
        env_id = self.cfg.task.split("-", 1)[-1] + "-v2"
        data_path = os.path.join(data_path, env_id)
        demo_list = os.listdir(data_path)

        for demo in demo_list:
            demo_path = os.path.join(data_path, demo)
            demo_actions = torch.tensor(np.load(os.path.join(demo_path, 'actions.npy')))
            demo_states = torch.tensor(np.load(os.path.join(demo_path, 'states.npy')))
            demo_rewards = torch.tensor(np.load(os.path.join(demo_path, 'rewards.npy')))
            rgbs = [np.load(os.path.join(demo_path, 'rgb', rgb)) for rgb in os.listdir(os.path.join(demo_path, 'rgb'))]
            rgbs = [torch.tensor(rgb,dtype=torch.uint8).permute(2,0,1).unsqueeze(0) for rgb in rgbs]
            with torch.no_grad():
                rgb_embedings = torch.stack([self.obs_encoder(rgb).to(self.save_device) for rgb in rgbs], dim=0)
            
            episode = Episode.from_trajectory(
                self.cfg, 
                len(rgb_embedings), 
                rgb_embedings, 
                demo_states,
                demo_actions,
                demo_rewards
                )
            setattr(episode, 'seed', demo)
            self.demo_episodes.append(episode)

            print(f'Encoding demostration from {demo_path}')

    def bc_sample(self):
        demo = random.choice(self.demo_episodes)
        if len(demo)<32:
            return (
                demo.obs_embeding.cuda(non_blocking=True),
                demo.state.cuda(non_blocking=True),
                demo.action.cuda(non_blocking=True)
                )
        index = np.random.randint(self.cfg.frame_stack-1, len(demo)-32)
        return (
            demo.obs_embeding[index:index+32].cuda(non_blocking=True),
            demo.state[index:index+32].cuda(non_blocking=True),
            demo.action[index:index+32].cuda(non_blocking=True)
        )
    
    def imagine_sample(self):
        p_episode = random.choice(self.practise_episodes)
        demo = self.demo_episodes[p_episode.demo_id]
        step = np.argmax(demo.value)    # 暂时用最简单的方法确定 imagine 步数
        return (
            p_episode.obs_embeding[:step].cuda(non_blocking=True), 
            p_episode.state[:step].cuda(non_blocking=True), 
            step, 
            p_episode.demo_similarity, 
            demo.obs_embeding[:step+1].cuda(non_blocking=True), 
            demo.state[:step+1].cuda(non_blocking=True)
            )
        
    
    def dynamilc_sample(self):
        '''
        dynamic 采样的时候，刻意避免了采样第一帧，
        因为在 frame_stack>1 的情况里，第一帧是重复的，而之后又是由连续多帧拼接的。两者不线性
        '''
        demo = random.choice(self.demo_episodes + self.practise_episodes)
        return (
            demo.obs_embeding[self.cfg.frame_stack-1:-1].cuda(non_blocking=True),
            demo.state[self.cfg.frame_stack-1:-1].cuda(non_blocking=True),
            demo.action[self.cfg.frame_stack-1:-1].cuda(non_blocking=True),
            demo.obs_embeding[self.cfg.frame_stack:].cuda(non_blocking=True),
            demo.state[self.cfg.frame_stack:].cuda(non_blocking=True),
        )

    def modem_get_demos(self, cfg):
        fps = glob.glob(str(Path(cfg.demo_dir) / "demonstrations" / f"{cfg.task}/*.pt"))
        for fp in fps:
            data = torch.load(fp)
            frames_dir = Path(os.path.dirname(fp)) / "frames"
            assert frames_dir.exists(), "No frames directory found for {}".format(fp)
            frame_fps = [frames_dir / fn for fn in data["frames"]]
            obss = [np.array(Image.open(fp)).transpose(2,1,0) for fp in frame_fps]
            obs_embedings = [torch.tensor(obs) for obs in obss[0:-1]]   
            with torch.no_grad():
                obs_embedings = torch.cat([self._obs_encoder(obs_embeding.unsqueeze(0)) for obs_embeding in obs_embedings],dim=0)
            
            obs_embeding = torch.empty(
                (obs_embedings.size()[0], self.cfg.frame_stack*self.cfg.latent_dim),
                dtype=torch.float32,
                device=self.save_device
            )
            for i in range(self.cfg.frame_stack):
                obs_embeding[self.cfg.frame_stack-1-i:,i*self.cfg.latent_dim:(i+1)*self.cfg.latent_dim]=obs_embedings[0:obs_embedings.size()[0]-self.cfg.frame_stack+1+i,:]
                for j in range(self.cfg.frame_stack-1-i):
                    obs_embeding[j,i*self.cfg.latent_dim:(i+1)*self.cfg.latent_dim]=obs_embedings[0]  # 读取demo时模仿env的frame—stack
            state = torch.tensor(data["states"], dtype=torch.float32)
            if cfg.task.startswith("mw-"):
                state = torch.cat((state[:, :4], state[:, 18 : 18 + 4]), dim=-1)
            elif cfg.task.startswith("adroit-"):
                if cfg.task == "adroit-door":
                    state = np.concatenate([state[:, :27], state[:, 29:32]], axis=1)
                elif cfg.task == "adroit-hammer":
                    state = state[:, :36]
                elif cfg.task == "adroit-pen":
                    state = np.concatenate([state[:, :24], state[:, -9:-6]], axis=1)
                else:
                    raise NotImplementedError()
            actions = np.array(data["actions"], dtype=np.float32).clip(-1, 1)
            if cfg.task.startswith("mw-") or cfg.task.startswith("adroit-"):
                rewards = (
                    np.array(
                        [
                            _data[
                                "success" if "success" in _data.keys() else "goal_achieved"
                            ]
                            for _data in data["infos"]
                        ],
                        dtype=np.float32,
                    )
                    - 1.0
                )
            else:  # use dense rewards for DMControl
                rewards = np.array(data["rewards"])
            episode = Episode.modem_from_trajectory(cfg, len(obs_embedings), obs_embeding, state[0:-1], actions, rewards)
            setattr(episode, 'seed', fp[-4])
            self.demo_episodes.append(episode)

            print(f'Encoding demostration from {fp}')




''' 世界模型
World Model:
    agent
    memory
        vlm
    dynamic model(dm)

    function:
        agent_bc
        dynamic model update
        agent update
'''

''' train
 初始化：
    创建日志文件
        
    创建记忆模型，编码专家轨迹
    创建世界模型
预训练：
    agent_bc
    dynamic model update
练习：
    for i in interaction_episodes:
        agent interaction in an episode
        memory model add episode
        memory update demostrition though one episode
    
    for i in train_freq:
        calculate dm_train_freq and imagine_train_freq through dm predict ability
        for j in dm_train_freq:
            train dynamic model
        for i in imagine_train_freq:
            train agent in imagine

关于求解奖励、每帧画面相似层度这部分，之前想的是用图像或者特征向量，今晚测试都不行。
不过可以考虑用 r3m 的奖励值的差异，我感觉可行，因为奖励值的曲线虽然不线性，但是每一组的差异层度不大。

'''