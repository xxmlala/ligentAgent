from agent import PolicyNet

import logging
import hydra
import torch
import numpy as np
import ligent
from omegaconf import OmegaConf
from dotmap import DotMap
from hydra.utils import instantiate
from pyvirtualdisplay import Display
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

    
def test_env(cfg, episodes, models_name):
    # ligent.set_scenes_dir("C:/Users/19355/Desktop/drlProject/LIGENT/custom_scenes")
    ligent.set_scenes_dir("./custom_scenes_test")
    env = ligent.Environment(path="/home/liuan/workspace/drl_project/ligent-linux-server/LIGENT.x86_64")
    env_decoder = ComeHereEnv(distance_reward=10, success_reward=200, distance_min=1.2, step_penalty=1, episode_len=100, is_debug=True)
    action_decoder = instantiate(cfg.action_decoder, device=device)
    
    feature_net = instantiate(cfg.feature_net, device=device)
    agent = instantiate(cfg.agent, preprocess_net=feature_net, device=device)
    # try: 
    agent.load_full_path(models_name, load_value_net=False)

    returns = []
    success_num = 0
    success_distance = 1.2
    eps = 1e-3
    for episode in range(episodes):
        print(episode, end=",", flush=True)
        state_img, _ = env.reset()
        state_text = torch.zeros(520)
        env_decoder.reset()
        blocked, done = False, False
        while not(done or blocked):
            action = agent.get_action((state_img, state_text), sample=False)
            action_env = action_decoder.decode(action)
            (state_img, state_text), reward, done, info = env.step(**action_env)
            reward, done, blocked, cumulate_reward, elspsed_step, distance_info = env_decoder.step(info)
        if distance_info[2] <= eps+success_distance:
            success_num += 1
        returns.append(cumulate_reward)
    print("test completed!")
    # except:
    env.close()
    ligent.set_scenes_dir("")
    print(f"models: {models_name}")
    print(returns)
    print(f"success_rate: {success_num/episodes}, mean reward: {sum(returns)/episodes}.")

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
# def main(cfg, episodes, models_name):
def main(cfg):
# def main(cfg):
    # log_dict = other_utils.get_log_dict(cfg.agent._target_)
    # for seed in cfg.seeds:
        # with torch.autograd.set_detect_anomaly(True):
    # train(cfg, seed, log_dict, logger, *(get_dataloader())) best_model_seed_3407_actor
    test_env(cfg, episodes=1000, models_name='/home/liuan/workspace/drl_project/ligentAgent/eval_models/directlyPPO/best_model_seed_3407_')
    # test_env(cfg, episodes=1000, models_name='/home/liuan/workspace/drl_project/ligentAgent/eval_models/IL_PPO/step_24576_model_seed_3407_')
    # test_env(cfg, episodes=1000, models_name='/home/liuan/workspace/drl_project/ligentAgent/eval_models/IL_PPO2/step_303104_model_seed_3407_')
    # start_time = time.time()
    # test_env(cfg, episodes=1000, models_name='/home/liuan/workspace/drl_project/ligentAgent/eval_models/IL/best_acc_')
    # print(f"It costs {time.time()-start_time} s!")
# @hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
# def main(cfg):
#     log_dict = other_utils.get_log_dict(cfg.agent._target_)
#     for seed in cfg.seeds:
#         with torch.autograd.set_detect_anomaly(True):
#             train(cfg, seed, log_dict, logger, *(get_dataloader()))
    


if __name__=="__main__":
    # main(episodes=5, models_name='best_acc_')
    with Display(visible=False) as disp:
        main()