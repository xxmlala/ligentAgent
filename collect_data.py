# import hydra
# import other_utils
from taskEnv import ComeHereEnv
# from agent.action_decoder import ActionDecoder
# from agent.features import FeatureFusion
# from agent.ppo import PPOAgent
# import torch
import ligent
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import math
import h5py
matplotlib.use('Agg')

def save_pic(np_arr, pic_path):
    fig, ax = plt.subplots()
    ax.imshow(np_arr)
    ax.axis('off')
    fig.set_size_inches(np_arr.shape[1]/100, np_arr.shape[0]/100)
    plt.savefig(pic_path,bbox_inches='tight', pad_inches=0, transparent=True) 
    plt.close()

def distance(a,b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5

def inference_action(info):
    position_player = info['game_states']['player']['position']
    position_mate = info['game_states']['playmate']['position']

    face_direction_player = info['game_states']['player']['forward']
    face_direction_mate = info['game_states']['playmate']['forward']

    mate2player = np.array([position_player['x']-position_mate['x'], position_player['z']-position_mate['z']])
    mate2player /= (mate2player[0]**2 + mate2player[1]**2)**0.5

    mate_direction = np.array([face_direction_mate['x'], face_direction_mate['z']])
    mate_direction /= (mate_direction[0]**2 + mate_direction[1]**2)**0.5

    def deg(v1,v2):
        # 计算向量的点积 
        dot_product = v1[0] * v2[0] + v1[1] * v2[1] # 计算向量的叉积 
        cross_product = v1[0] * v2[1] - v1[1] * v2[0] # 计算带符号夹角（弧度） 
        angle = math.atan2(cross_product, dot_product)
        ret_degree = angle/math.pi * 180
        # print(degree)
        return ret_degree
    
    degree = deg(mate2player, mate_direction)
    # inner_product = lambda a,b: max(min(a[0]*b[0]+a[1]*b[1], 1), -1)
    # outer_product = lambda a,b: a[0]*b[1]-a[1]*b[0]
    # degree = math.acos(inner_product(mate_direction, mate2player)) / math.pi * 180

    action_encoder = {
        'no_op': 0,
        'move_forward': 1,
        'look_yaw_n30d': 2,
        'look_yaw_p30d': 3,
        'look_pitch_n30d': 4,
        'look_pitch_p30d': 5,
        'is_grab': 6,
        'is_speak': 7
    }

    if abs(degree) <= 15:
        action_env = {
            "move_right": 0,
            "move_forward": 1,
            "look_yaw": 0.0,
            "look_pitch": 0.0,
            "jump": False,
            "grab": False,
            "speak": "",
            }
        action_str = 'move_forward'
    elif degree>=-30 and degree<-15:
        action_env = {
            "move_right": 0,
            "move_forward": 0,
            "look_yaw": -30,
            "look_pitch": 0.0,
            "jump": False,
            "grab": False,
            "speak": "",
            }
        action_str = "look_yaw_n30d"
    else:
        action_env = {
            "move_right": 0,
            "move_forward": 0,
            "look_yaw": 30,
            "look_pitch": 0.0,
            "jump": False,
            "grab": False,
            "speak": "",
            }
        action_str = 'look_yaw_p30d'
    
    return action_env, action_encoder[action_str], action_str

def collect(env,episodes:int, instruction:str="come here"):
    env_decoder = ComeHereEnv(distance_reward=10, success_reward=200, distance_min=1.2, step_penalty=1, episode_len=100, is_debug=True)

    action_noop = {
            "move_right": 0,
            "move_forward": 0,
            "look_yaw": 0.0,
            "look_pitch": 0.0,
            "jump": False,
            "grab": False,
            "speak": "",
            }
    returns = []

    if not os.path.exists('./dataset'):
        os.makedirs('./dataset')

    with h5py.File('./dataset/Episode1000.h5', 'w') as f:
        data_nums = 0
        fail_nums = 0
        success_episode, fail_episode = 0,0
        obs_V_set = f.create_dataset('obs_V', (0,224,224,3), maxshape=(None, 224, 224, 3), dtype='i')
        obs_T_set = f.create_dataset('obs_T', (0,), maxshape=(None,),dtype=h5py.string_dtype("utf-8")) #TODO 往出拿时需要 “ss”.decode("utf-8")
        action_set = f.create_dataset('action', (0,), maxshape=(None,),dtype='i')
        # data_obs_V = []
        # data_obs_T = []
        # data_action = []
        for episode in range(episodes):
            if episode%5==0:
                print(f"Episode: {episode}/{episodes}")
            path = f"./obs_visions/episode_{episode}"
            if not os.path.exists(path):
                os.makedirs(path)
            state_img, _ = env.reset()
            env_decoder.reset()
            done, blocked = False, False
            (state_img, state_text), _, _, info = env.step(**action_noop)
            
            obs_list = []

            while not (done or blocked):
                last_state_img = state_img
                last_state_text = state_text
                last_state_text = instruction
                action_env, action_code, action_str = inference_action(info)
                (state_img, state_text), _, _, info = env.step(**action_env)
                reward, done, blocked, cumulate_reward, elspsed_step, distance_info = env_decoder.step(info)
                obs_list.append((last_state_img, last_state_text, action_code, path+f'/{elspsed_step:03d}_{action_str}_{round(reward,1)}.png'))

            obs_list.append((state_img, instruction, 0, path+f'/{elspsed_step+1:03d}_noOp_NULL.png'))
            if blocked:
                fail_nums += len(obs_list)
                fail_episode += 1
                print(f"BLOCKED_sum: {fail_nums}")
                continue
            success_episode += 1
            for state_img,state_text,action_code,pic_path in obs_list:
                data_nums += 1
                obs_V_set.resize((data_nums, 224,224,3))
                obs_T_set.resize((data_nums,))
                action_set.resize((data_nums,))
                obs_V_set[data_nums-1] = state_img
                obs_T_set[data_nums-1] = state_text
                action_set[data_nums-1] = action_code
                save_pic(state_img, pic_path=pic_path)

            os.rename(path, path + f"_{elspsed_step}_{distance_info}_{cumulate_reward}")
    print(returns)
    env.close()
    print(f"{success_episode} success_episodes, {fail_episode} fail_episodes, {data_nums} data pairs.")
    
def debug(env):
    action_noop = {
            "move_right": 0,
            "move_forward": 0,
            "look_yaw": 0.0,
            "look_pitch": 0.0,
            "jump": False,
            "grab": False,
            "speak": "",
            }
    action_right15 = {
            "move_right": 0,
            "move_forward": 1,
            "look_yaw": 0,
            "look_pitch": 0.0,
            "jump": False,
            "grab": False,
            "speak": "",
            }
    ligent.set_scenes_dir("C:/Users/19355/Desktop/drlProject/LIGENT/custom_scenes")
    # env = ligent.Environment(path="C:/Users/19355/Desktop/drlProject/05272014_fix_multi_rotate/305272014_fix_multi_rotate/LIGENT.exe")
    env_decoder = ComeHereEnv(distance_reward=10, success_reward=200, distance_min=1.2, step_penalty=1, episode_len=100, is_debug=True)
    (state_img, _), _, _, info = env.step(**action_noop)
    print(info['game_states']['playmate']['forward'], flush=True)
    last_forward = info['game_states']['playmate']['forward']
    last_position = info['game_states']['playmate']['position']
    last_time = time.time()
    for step in range(360//15):
        (state_img, _), _, _, info = env.step(**action_right15)
        print(f"Cost {time.time()-last_time} s!")
        last_time = time.time()
        # reward, done, blocked, cumulate_reward, elspsed_step, distance_info = env_decoder.step(info)
        print(info['game_states']['playmate']['forward'], flush=True)
        current_forward = info['game_states']['playmate']['forward']
        current_position = info['game_states']['playmate']['position']
        print(degree([last_forward['x'],last_forward['z']],[current_forward['x'],current_forward['z']]))
        print(distance([current_position['x'],current_position['z']],[last_position['x'],last_position['z']]))
        # time.sleep(3)
        last_forward = current_forward
        last_position = current_position

    env.close()
    ligent.set_scenes_dir("")

if __name__ == "__main__":
    

    ligent.set_scenes_dir("")
    env = ligent.Environment(path="C:/Users/19355/Desktop/drlProject/ligent-windows/06061943_win_224/LIGENT.exe")
    collect(env,1000)
    exit(0)
    # with Display(visible=False) as disp:
    try:
        ligent.set_scenes_dir("")
        env = ligent.Environment(path="C:/Users/19355/Desktop/drlProject/ligent-windows/06061943_win_224/LIGENT.exe")
        collect(env,10)
        # debug(env)
    except:
        print("Some error happened!")
        env.close()
    # debug()