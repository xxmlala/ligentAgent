import logging
from copy import deepcopy
from typing import Optional

import torch
import numpy as np
import ligent
from torch import multiprocessing as mp
from omegaconf import OmegaConf
from dotmap import DotMap
from hydra.utils import instantiate
# from gymnasium.wrappers import RecordEpisodeStatistics, ClipAction, \
#     NormalizeObservation, TransformObservation, NormalizeReward, \
#     TransformReward, RecordVideo
# from gym.wrappers import RecordVideo
import other_utils
from agent.ppo import PPOAgent
from buffer import ReplayBuffer, PrioritizedReplayBuffer, PPOReplayBuffer, get_buffer

from taskEnv import ComeHereEnv
import time
'''
only modified the PPO related Buffer (ReplayBuffer and PPOBuffer)

'''



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cpu'
# TODO: the return value in vectorized env is significantly lower than the non-vectorized env, why?
def eval(env, agent, episodes, seed, action_decoder, env_decoder):
    returns = []
    distance_info_s = []
    ligent.set_scenes_dir("./custom_scenes")
    # eval_start = time.time()
    # print("Start eval!", flush=True)
    for episode in range(episodes):
        # state, _ = env.reset(seed=np.random.randint(0, 10000) + seed)
        state_img, state_text = env.reset()
        env_decoder.reset()
        done, blocked = False, False
        while not (done or blocked):
            state_img = np.expand_dims(state_img, 0)
            state_text = np.expand_dims(state_text, 0)
            action = agent.get_action((state_img,state_text), sample=False)
            action_env = action_decoder.decode(action)
            # state, _, _, info = env.step(action_env)
            (state_img, state_text), _, _, info = env.step(**action_env)
            reward, done, blocked, cumulate_reward, elpased_step, distance_info = env_decoder.step(info)
        # returns.append(info['episode']['r'].item())
        returns.append(cumulate_reward)
        distance_info_s.append(distance_info)
    # print(f"eval costs {time.time()-eval_start} s!", flush=True)
    ligent.set_scenes_dir("")
    return np.mean(returns), np.std(returns), distance_info_s


def train(cfg, seed: int, log_dict: dict, idx: int, logger: logging.Logger, barrier: Optional[mp.Barrier]):
    # make_env = lambda: TransformReward(
    #     NormalizeReward(
    #         TransformObservation(
    #             NormalizeObservation(
    #                 ClipAction(
    #                     RecordEpisodeStatistics(
    #                         gym.make(cfg.env_name, render_mode="rgb_array")
    #                     )
    #                 )
    #             ), lambda obs: np.clip(obs, -10, 10)
    #         )
    #     ), lambda reward: np.clip(reward, -10, 10)
    # )
    # env = gym.vector.SyncVectorEnv([make_env] * cfg.vec_envs) if cfg.vec_envs > 1 else make_env()
    env = ligent.Environment(path="/home/liuan/workspace/drl_project/ligent-linux-server/LIGENT.x86_64")
    env_decoder = ComeHereEnv(distance_reward=10, success_reward=200, distance_min=1.2, step_penalty=1, episode_len=500, is_debug=True)
    action_decoder = instantiate(cfg.action_decoder, device=device)
    other_utils.set_seed_everywhere(env, seed)

    eval_env_decoder = ComeHereEnv(distance_reward=10, success_reward=200, distance_min=1.2, step_penalty=1, episode_len=100, is_debug=True)
    # state_size = other_utils.get_space_shape(env.observation_space, is_vector_env=cfg.vec_envs > 1)
    # action_size = other_utils.get_space_shape(env.action_space, is_vector_env=cfg.vec_envs > 1)

    # buffer = get_buffer(cfg.buffer, device=device, seed=seed)

    buffer = instantiate(cfg.buffer, device=device, seed=seed)
    feature_net = instantiate(cfg.feature_net, device=device)
    agent = instantiate(cfg.agent, preprocess_net=feature_net, device=device)
                        # action_space=env.action_space if cfg.vec_envs <= 1 else env.envs[0].action_space, device=device)

    # get_attr of omega_conf is slow, so we convert it to dotmap
    cfg = DotMap(OmegaConf.to_container(cfg.train, resolve=True))

    logger.info(f"Training seed {seed} for {cfg.timesteps} timesteps with {agent} and {buffer}")

    using_mp = barrier is not None
    
    if using_mp:
        local_log_dict = {key: [] for key in log_dict.keys()}
    else:
        local_log_dict = log_dict
        for key in local_log_dict.keys():
            local_log_dict[key].append([])
    
    done, blocked, best_reward = False, False, -np.inf
    if cfg.vec_envs > 1:
        done, truncated = np.array([False] * cfg.vec_envs), np.array([False] * cfg.vec_envs)

    # state, _ = env.reset(seed=seed)
    state = env.reset()
    env_decoder.reset()
    last_reset_time = time.time()
    just_evaluated = False
    for step in range(cfg.vec_envs, cfg.timesteps + 1, cfg.vec_envs):
        # print(f"step_{step}", flush=True)
        if cfg.vec_envs > 1 and done.any():
            rewards = np.array([d['episode']['r'] for d in info['final_info'][info['_final_info']]]).squeeze(-1)
            other_utils.write_to_dict(local_log_dict, 'train_returns', np.mean(rewards).item(), using_mp)
            other_utils.write_to_dict(local_log_dict, 'train_steps', step - cfg.vec_envs, using_mp)
        elif cfg.vec_envs <= 1 and (done or just_evaluated or blocked):
            just_evaluated = False
            state = env.reset()
            env_decoder.reset()
            print(f"It gets {round(cumulate_reward,3)} sum reward, costs {elspsed_step} steps, distance_info: {distance_info}")
            last_reset_time = time.time()
            done, blocked = False, False
            # other_utils.write_to_dict(local_log_dict, 'train_returns', info['episode']['r'].item(), using_mp)
            other_utils.write_to_dict(local_log_dict, 'train_returns', round(cumulate_reward/elspsed_step,3), using_mp)
            other_utils.write_to_dict(local_log_dict, 'train_steps', step - 1, using_mp)

        if isinstance(agent, PPOAgent):
            action, log_prob = agent.act(state, sample=True)
        else:
            action = agent.get_action(state, sample=True)
        
        action_env = action_decoder.decode(action)
        # next_state, reward, done, truncated, info = env.step(action)
        next_state, reward, done, info = env.step(**action_env)
        reward, done, blocked, cumulate_reward, elspsed_step, distance_info = env_decoder.step(info)
        if isinstance(buffer, PPOReplayBuffer):
            value = agent.get_value(state)
            if cfg.vec_envs > 1 and done.any(): # won't be exectued
                idxs, = info['_final_observation'].nonzero()
                next_state[idxs] = np.vstack(info['final_observation'][idxs])
            buffer.add((state, action, reward, next_state, done, value, log_prob))
        else:
            buffer.add((state, action, reward, next_state, int(done)))
        state = next_state

        if step >= cfg.batch_size + cfg.nstep:
            if isinstance(buffer, PrioritizedReplayBuffer):
                batch, weights, tree_idxs = buffer.sample(cfg.batch_size)
                ret_dict = agent.update(batch, weights=weights)
                buffer.update_priorities(tree_idxs, ret_dict['td_error'])

            elif isinstance(buffer, ReplayBuffer) or isinstance(buffer, PPOReplayBuffer):
                if isinstance(agent, PPOAgent):
                    # update PPO only if the buffer is full
                    if not (step) % cfg.ppo_update_interval:
                        buffer.compute_advantages_and_returns(agent)
                        ret_dict = agent.update(buffer)
                        buffer.clear()
                    else:
                        ret_dict = {}
                else:
                    batch = buffer.sample(cfg.batch_size)
                    ret_dict = agent.update(batch)

            else:
                raise RuntimeError("Unknown buffer")

            for key in ret_dict.keys():
                other_utils.write_to_dict(local_log_dict, key, ret_dict[key], using_mp)

        eval_cond = step % cfg.eval_interval == 0
        if cfg.vec_envs > 1:
            eval_cond = step > cfg.vec_envs + 1 and np.any(np.arange(step - cfg.vec_envs + 1, step + 1) % cfg.eval_interval == 0)
        if eval_cond:
            eval_mean, eval_std, distance_info_s = eval(env, agent=agent, episodes=cfg.eval_episodes, seed=seed, 
                                       action_decoder=action_decoder, env_decoder=eval_env_decoder)
            other_utils.write_to_dict(local_log_dict, 'eval_steps', step - 1, using_mp)
            other_utils.write_to_dict(local_log_dict, 'eval_returns', eval_mean, using_mp)
            logger.info(f"Seed: {seed}, Step: {step}, Eval mean: {eval_mean:.2f}, Eval std: {eval_std:.2f}")
            logger.info(f"distances_info: {distance_info_s}")
            if eval_mean > best_reward:
                best_reward = eval_mean
                if using_mp:
                    logger.info(f'Seed: {seed}, Save best model at eval mean {best_reward:.4f} and step {step}')
                else:
                    logger.info(f'Seed: {seed}, Save best model at eval mean {best_reward:.4f} and step {step}')
                agent.save(f'best_model_seed_{seed}_')
            just_evaluated = True
            
        
        plot_cond = step % cfg.plot_interval == 0
        if cfg.vec_envs > 1:
            plot_cond = step > cfg.vec_envs + 1 and np.any(np.arange(step - cfg.vec_envs, step) % cfg.plot_interval == 0)
        if plot_cond:
            other_utils.sync_and_visualize(log_dict, local_log_dict, barrier, idx, step, f'{agent} with {buffer}', using_mp)
        if step%8192==0:
            agent.save(f'step_{step}_model_seed_{seed}_')
            logger.info(f'Seed: {seed}, Save step_{step} model!')
    agent.save(f'final_model_seed_{seed}_')
    other_utils.sync_and_visualize(log_dict, local_log_dict, barrier, idx, step, f'{agent} with {buffer}', using_mp)

    # env = RecordVideo(eval_env, f'final_videos_seed_{seed}', name_prefix='eval', episode_trigger=lambda x: x % 3 == 0 and x < cfg.eval_episodes, disable_logger=True)
    # eval_mean, eval_std = eval(env, agent=agent, episodes=cfg.eval_episodes, seed=seed)

    # agent.load(f'best_model_seed_{seed}.pt')  # use best model for visualization
    # env = RecordVideo(eval_env, f'best_videos_seed_{seed}', name_prefix='eval', episode_trigger=lambda x: x % 3 == 0 and x < cfg.eval_episodes, disable_logger=True)
    # eval_mean, eval_std = eval(env, agent=agent, episodes=cfg.eval_episodes, seed=seed)
    # other_utils.merge_videos(f'final_videos_seed_{seed}')
    # other_utils.merge_videos(f'best_videos_seed_{seed}')
    # env.close()
    # logger.info(f"Finish training seed {seed} with everage eval mean: {eval_mean}")
    # return eval_mean
    return eval_mean
