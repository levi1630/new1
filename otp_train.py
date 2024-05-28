import os
import torch
import numpy as np
import time
import hydra

from env import make_env, set_seed
from otp import OTP
from pathlib import Path
from cprint import cprint

import tools as t
import logger


def evaluate(env, agent, cfg, env_step=None, video=False):
    """Evaluate a trained agent and optionally save a video."""
    episode_rewards = []
    episode_successes = []
    for i in range(cfg.eval_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        if video:
            video.init(env, enabled=(i == 0))
        while not done:
            action = agent.get_action_from_obs(obs, env.state, eval_mode=True)
            obs, reward, done, info = env.step(action.cpu().numpy())
            ep_reward += reward
            if video:
                video.record(env)
            t += 1
        episode_rewards.append(ep_reward)
        episode_successes.append(info.get("success", 0))
        if video:
            video.save(env_step)
    return np.nanmean(episode_rewards), np.nanmean(episode_successes)

@hydra.main(config_name='otp_config', config_path='cfgs')
def train(cfg:dict):
    '''Training OTP online on one task'''

    # Initialization
    assert torch.cuda.is_available()
    set_seed(cfg.seed)
    work_dir = Path(cfg.logging_dir) / cfg.task / str(cfg.seed)
    cprint.info(f"Work dir: {work_dir}")
    env, otp = make_env(cfg), OTP(cfg)
    # L = logger.Logger(work_dir, cfg)
    t.model_structure(otp.agent)

    # otp pretrain
    if cfg.behaviour_clone:
        for i in range(1,2):
            otp.agent_bc()
            rewards, sr = evaluate(env, otp, cfg)
            cprint(f'''
                   bc_steps: {i * cfg.bc_iteration}
                    success_rate: {sr}
                    reward: {rewards}''')
            if sr > cfg.bc_threshold:
                cprint.ok(f'Behaviour clone finished!')
                break

    otp.dynamic_model_update()
    
    # practise online
    for step in range(cfg.iteration_episode):
        
        # interact with env
        obs = env.reset()
        if cfg.DEBUG: t.save_rgb_array_as_jpg(os.getcwd()+'/debug', 'init_rgb', obs[0:3])
        episode, obs_embeding = otp.memory.init_episode(obs, env.state)
        while not (episode.done or episode.truncated):
            with torch.no_grad():
                action = otp.get_action_from_obs(obs, env.state)
                obs, reward, done, info = env.step(action.squeeze(0).cpu().numpy())
                obs_embeding = otp.memory.obs_encoder(obs)
                episode += (obs_embeding, env.state, action, reward, done)

        otp.memory += episode

        # update dynamic model
        otp.dynamic_model_update()

        # update agent 
        otp.agent_update()

        # log training data

        # evaluate agent
        if step % cfg.eval_freq == 0:
            eval_rew, eval_succ = evaluate(env, otp, cfg, False)
      
    # L.finish()
if __name__ == "__main__":
    train()

