from agent import PolicyNet

import logging
import hydra
import torch
import numpy as np
import ligent
from omegaconf import OmegaConf
from dotmap import DotMap
from hydra.utils import instantiate
# from pyvirtualdisplay import Display
import other_utils
import time
from taskEnv import ComeHereEnv

logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision('high')
# device='cpu'
def eval_env(env, agent, episodes, seed, action_decoder, env_decoder):
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

def train(cfg, seed: int, log_dict: dict, logger: logging.Logger, train_loader, eval_loader):
    # env = ligent.Environment(path="/home/liuan/workspace/drl_project/ligent-linux-server/LIGENT.x86_64")
    # env_decoder = ComeHereEnv(distance_reward=10, success_reward=200, distance_min=1.2, step_penalty=1, episode_len=500, is_debug=True)
    # action_decoder = instantiate(cfg.action_decoder, device=device)
    other_utils.set_seed_everywhere("", seed)
    # eval_env_decoder = ComeHereEnv(distance_reward=10, success_reward=200, distance_min=1.2, step_penalty=1, episode_len=100, is_debug=True)
    # state_size = other_utils.get_space_shape(env.observation_space, is_vector_env=cfg.vec_envs > 1)
    # action_size = other_utils.get_space_shape(env.action_space, is_vector_env=cfg.vec_envs > 1)

    feature_net = instantiate(cfg.feature_net, device=device)
    
    agent = instantiate(cfg.agent, preprocess_net=feature_net, device=device)
    cfg = DotMap(OmegaConf.to_container(cfg.train, resolve=True))
    agent.load('')
    # model = PolicyNet(feature_net=agent.get_feature_net(), actor_net=agent.get_actor_net())
    
def test_env(cfg, episodes, models_name):
    # ligent.set_scenes_dir("C:/Users/19355/Desktop/drlProject/LIGENT/custom_scenes")
    env = ligent.Environment(path="C:/Users/19355/Desktop/drlProject/05272014_fix_multi_rotate/305272014_fix_multi_rotate/LIGENT.exe")
    env_decoder = ComeHereEnv(distance_reward=10, success_reward=200, distance_min=1.2, step_penalty=1, episode_len=100, is_debug=True)
    action_decoder = instantiate(cfg.action_decoder, device=device)
    
    feature_net = instantiate(cfg.feature_net, device=device)
    agent = instantiate(cfg.agent, preprocess_net=feature_net, device=device)
    # try: 
    agent.load_in_windows(models_name)
    for episode in range(episodes):
        state_img, _ = env.reset()
        state_text = torch.zeros(520)
        env_decoder.reset()
        blocked, done = False, False
        while not(done or blocked):
            action = agent.get_action((state_img, state_text), sample=False)
            action_env = action_decoder.decode(action)
            (state_img, state_text), reward, done, info = env.step(**action_env)
            reward, done, blocked, cumulate_reward, elspsed_step, distance_info = env_decoder.step(info)
    # except:
    env.close()


@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
# def main(cfg, episodes, models_name):
def main(cfg):
# def main(cfg):
    # log_dict = other_utils.get_log_dict(cfg.agent._target_)
    # for seed in cfg.seeds:
        # with torch.autograd.set_detect_anomaly(True):
    # train(cfg, seed, log_dict, logger, *(get_dataloader()))
    test_env(cfg, episodes=10, models_name='C:/Users/19355/Desktop/drlProject/ligentAgent/models/best_acc_')
    # test_env(cfg, episodes=5, models_name='best_acc_')

# @hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
# def main(cfg):
#     log_dict = other_utils.get_log_dict(cfg.agent._target_)
#     for seed in cfg.seeds:
#         with torch.autograd.set_detect_anomaly(True):
#             train(cfg, seed, log_dict, logger, *(get_dataloader()))
    


if __name__=="__main__":
    # main(episodes=5, models_name='best_acc_')
    main()