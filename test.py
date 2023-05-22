'''
test in windows os
'''


from taskEnv import ComeHereEnv
from agent.action_decoder import ActionDecoder
from agent.features import FeatureFusion
from agent.ppo import PPOAgent
import torch
import ligent
import numpy as np

# logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision('high')

def test(episodes:int):
    env = ligent.Environment(path="C:/Users/19355/Desktop/drlProject/05222130_release_5656/LIGENT.exe")
    env_decoder = ComeHereEnv(distance_reward=10, distance_min=1.2, episode_len=500)
    # action_decoder = instantiate(cfg.action_decoder, device=device)
    action_decoder = ActionDecoder(decode_idx = 6, action_used_dims=7, action_env_dims=520, device=device)
    feature_net = FeatureFusion(hidden_depth=2, text_input_size=520, text_hidden_size=10,
                                img_input_size=[56,56,3], img_hidden_size=128, device=device)
    agent = PPOAgent(preprocess_net=feature_net, action_dims=[3, 4, 13, 13, 2, 2, 2], hidden_dim=400, hidden_depth=3, gamma=0.99,
                     tau=0.005, lr=3e-4, clip_range=0.2, value_clip_range=None, value_coef=1,
                     entropy_coef=0.01, update_epochs=10, mini_batch_size=512, device=device, nstep=1)
    agent.load("model_seed_3407.pt")
    returns = []
    for episode in range(episodes):
        # state, _ = env.reset(seed=np.random.randint(0, 10000) + seed)
        state_img, state_text = env.reset()
        env_decoder.reset()
        done, truncated = False, False
        while not (done or truncated):
            state_img = np.expand_dims(state_img, 0)
            state_text = np.expand_dims(state_text, 0)
            action = agent.get_action((state_img,state_text), sample=False).squeeze(0)
            action_env = action_decoder.decode(action)
            # state, _, _, info = env.step(action_env)
            (state_img, state_text), _, _, info = env.step(**action_env)
            reward, done, cumulate_reward, elpased_step = env_decoder.step(info)
        # returns.append(info['episode']['r'].item())
        returns.append(cumulate_reward/elpased_step)

if __name__ == "__main__":
    
    # with Display(visible=False) as disp:
    test(5)
