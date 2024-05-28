# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''
第一版：用 r3m 表征 不用plan  调整不同 latent 大小
第二版：加上 单独的表征网络 可以输入 r3m 的结果 或者 原始图像
第三版：加上 plan 
'''
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
from PIL import Image
from pathlib import Path
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal

from memory import Memory, cal_similarity

__REDUCE__ = lambda b: "mean" if b else "none"


def l1(pred, target, reduce=False):
    """Computes the L1-loss between predictions and targets."""
    return F.l1_loss(pred, target, reduction=__REDUCE__(reduce))


def mse(pred, target, reduce=False):
    """Computes the MSE loss between predictions and targets."""
    return F.mse_loss(pred, target, reduction=__REDUCE__(reduce))

def calculate_correctness(pred, target):
    return


def dynamics(in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
    """Returns an MLP."""
    if isinstance(mlp_dim, int):
        mlp_dim = [mlp_dim, mlp_dim]
    return nn.Sequential(
        nn.Linear(in_dim, mlp_dim[0]),
        act_fn,
        nn.Linear(mlp_dim[0], mlp_dim[1]),
        act_fn,
        nn.Linear(mlp_dim[1], out_dim),
    )

class Dynamics(nn.Module):
    def __init__(self, in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
        super().__init__()
        if isinstance(mlp_dim, int):
            mlp_dim = [mlp_dim, mlp_dim]
        
        self.layers = nn.Sequential(
            nn.Linear(in_dim, mlp_dim[0]),
            act_fn,
            nn.Linear(mlp_dim[0], mlp_dim[1]),
            act_fn,
            nn.Linear(mlp_dim[1], out_dim),
        )

    def forward(self, x):
        return self.layers(x)
    
    def predict(self, obs_embedings, states, actions):
        dynamic_input = torch.cat((obs_embedings, states, actions), dim=1)
        return self.forward(dynamic_input)
    

def agent(in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
    """Returns an MLP."""
    if isinstance(mlp_dim, int):
        mlp_dim = [mlp_dim, mlp_dim]
    return nn.Sequential(
        nn.Linear(in_dim, mlp_dim[0]),
        act_fn,
        nn.Linear(mlp_dim[0], mlp_dim[1]),
        act_fn,
        nn.Linear(mlp_dim[1], out_dim),
    )

class Agent(nn.Module):
    def __init__(self, in_dim, mlp_dim, out_dim, act_fn=nn.ELU()):
        super().__init__()
        if isinstance(mlp_dim, int):
            mlp_dim = [mlp_dim, mlp_dim]
        
        self.layers = nn.Sequential(
            nn.Linear(in_dim, mlp_dim[0]),
            act_fn,
            nn.Linear(mlp_dim[0], mlp_dim[1]),
            act_fn,
            nn.Linear(mlp_dim[1], out_dim),
        )

    def forward(self, x):
        return self.layers(x)
    
    def get_action(self, obs_embeding, state):
        agent_input = torch.cat((obs_embeding, state), dim=1)
        return self.forward(agent_input)

class TruncatedNormal(pyd.Normal):
    """Utility class implementing the truncated normal distribution."""

    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class OTP(nn.Module):

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        self.device = torch.device('cuda')
        self._dynamics = Dynamics(cfg.latent_dim*self.cfg.frame_stack + cfg.state_dim + cfg.action_dim, 
                                  cfg.mlp_dim*self.cfg.frame_stack, 
                                  cfg.latent_dim*self.cfg.frame_stack + cfg.state_dim
                                  ).to(self.device)
        self.agent = Agent(cfg.latent_dim*self.cfg.frame_stack + cfg.state_dim, 
                           cfg.mlp_dim*self.cfg.frame_stack, 
                           cfg.action_dim
                           ).to(self.device)
        self.memory = Memory(self.cfg)
        self.agent_optim = torch.optim.Adam(self.agent.parameters(), lr=self.cfg.lr)
        self._dynamics_optim = torch.optim.Adam(self._dynamics.parameters(), lr=self.cfg.lr)
      
    def pi(self, z, std=0):
        """Samples an action from the learned policy (pi)."""
        mu = torch.tanh(self._pi(z))
        if std > 0:
            std = torch.ones_like(mu) * std
            return TruncatedNormal(mu, std).sample(clip=0.3)
        return mu
    
    def get_action_from_obs(self, obs, state, eval_mode=False):
        with torch.no_grad():
            obs_embeding = self.memory.obs_encoder(obs)
        state = torch.tensor(state).unsqueeze(0).to(device=self.device)
        if eval_mode:
            with torch.no_grad():
                return self.agent.get_action(obs_embeding, state).squeeze(0)
        return self.agent.get_action(obs_embeding, state).squeeze(0)

    def agent_bc(self):
        self.agent.train()
        for _ in range(self.cfg.bc_iteration):
            obs_embeding, state, action = self.memory.bc_sample()
            self.agent_optim.zero_grad(set_to_none=True)
            a = self.agent.get_action(obs_embeding, state)
            bc_loss=mse(a, action, reduce=True)
            bc_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.agent.parameters(),
                self.cfg.grad_clip_norm,
                error_if_nonfinite=False,
            )
            self.agent_optim.step()

        print(f'bc_loss: {bc_loss}')

    def _save_bc_agent(self):
        pass

    def dynamic_model_update(self):
        self._dynamics.train()
        for _ in range(self.cfg.bc_iteration * 2):
            obs_embedings, states, actions, nobs_embedings, nstates = self.memory.dynamilc_sample()
            self._dynamics_optim.zero_grad(set_to_none=True)
            next_embedings = torch.cat((nobs_embedings, nstates), dim=1)
            predicta_embedings = self._dynamics.predict(obs_embedings, states, actions)
            loss = mse(predicta_embedings, next_embedings, reduce=True)
            torch.nn.utils.clip_grad_norm_(
                self.agent.parameters(),
                self.cfg.grad_clip_norm,
                error_if_nonfinite=False,
            )
            self._dynamics_optim.step()

            print(f'bc_loss: {loss}')

    def agent_update(self):
        for _ in range(self.cfg.agent_update_iteration):
            obs_embedings, state, step, similarity, demo_embeding, demo_state = self.memory.imagine_sample()
            input_embeding  = torch.cat((obs_embedings[0].unsqueeze(0), state[0].unsqueeze(0)), dim=1)
            embeding = torch.cat((demo_embeding, demo_state), dim = 1)
            loss = []
            
            for i in range(step):
                a = self.agent(input_embeding)
                next_embeding = self._dynamics(torch.cat((input_embeding, a), dim=1))
                input_embeding = next_embeding
                loss.append(cal_similarity(next_embeding, embeding[i].detach()) * similarity)

            dynamics_loss =  sum(loss)
            self.agent_optim.zero_grad(set_to_none=True)
            dynamics_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(
                self.agent.parameters(),
                self.cfg.grad_clip_norm,
                error_if_nonfinite=False,
            )
            self.agent_optim.step()
        



