
from taskEnv import ComeHereEnv
from ligent.server.server import set_object_counts
import ligent
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import math
import h5py
import argparse

matplotlib.use('Agg')

def save_pic(np_arr, pic_path):
    fig, ax = plt.subplots()
    ax.imshow(np_arr)
    ax.axis('off')
    fig.set_size_inches(np_arr.shape[1]/100, np_arr.shape[0]/100)
    plt.savefig(pic_path,bbox_inches='tight', pad_inches=0, transparent=True) 
    plt.close()


def inference_come_to_sth_action(info, goal_object:str='player'):
    position_mate = info['game_states']['playmate']['position']
    position_camera_mate = info['game_states']['playmateCamera']['position']
    face_direction_camera_mate = info['game_states']['playmateCamera']['forward']
    face_direction_mate = info['game_states']['playmate']['forward']

    if goal_object=='player':
        position_goal = info['game_states']['player']['position']
        face_direction_player = info['game_states']['player']['forward']
        position_camera_player = info['game_states']['playerCamera']['position']

        x2z_norm = ((position_camera_player['x']-position_camera_mate['x'])**2 + \
                    (position_camera_player['z']-position_camera_mate['z'])**2)**0.5
        camera_mate2goal = np.array([position_camera_player['y']-position_camera_mate['y'], x2z_norm])
    else:
        instances_list = [i['prefab'] for i in info['game_states']['instances']]
        goal_object_ids = []
        for i,instance in enumerate(instances_list):
            # if instance==goal_object: #Pumpkin(Clone) == Pumpkin
            if goal_object in instance:
                goal_object_ids.append(i)
        assert len(goal_object_ids)==1,f"len(collect_object_idx) is {len(goal_object_ids)}"
        goal_object_id = goal_object_ids[0]
        position_goal = info['game_states']['instances'][goal_object_id]['position']
        face_direction_player = info['game_states']['instances'][goal_object_id]['forward']

        x2z_norm = ((position_goal['x']-position_camera_mate['x'])**2 + \
                    (position_goal['z']-position_camera_mate['z'])**2)**0.5
        camera_mate2goal = np.array([position_goal['y']-position_camera_mate['y'], x2z_norm])
    
    def vector_norm(v):
        assert len(v)==2, len(v)
        return (v[0]**2+v[1]**2)**0.5

    def deg(v1, v2):
        v1 = v1/vector_norm(v1)
        v2 = v2/vector_norm(v2)
        # 计算向量的点积 
        dot_product = v1[0] * v2[0] + v1[1] * v2[1] # 计算向量的叉积 
        cross_product = v1[0] * v2[1] - v1[1] * v2[0] # 计算带符号夹角（弧度） 
        angle = math.atan2(cross_product, dot_product)
        ret_degree = angle/math.pi * 180
        # print(degree)
        return ret_degree


    mate2goal = np.array([position_goal['x']-position_mate['x'], position_goal['z']-position_mate['z']])
    face_direction_mate = np.array([face_direction_mate['x'], face_direction_mate['z']])
    x2z_norm = (face_direction_camera_mate['x']**2 + face_direction_camera_mate['z']**2)**0.5
    face_direction_camera_mate = np.array([face_direction_camera_mate['y'],x2z_norm])

    degree_yaw = deg(mate2goal, face_direction_mate)
    degree_pitch = deg(camera_mate2goal, face_direction_camera_mate)

    # action_encoder = {
    #     'no_op': 0,
    #     'move_forward': 1,
    #     'look_yaw_n30d': 2,
    #     'look_yaw_n8d': 3,
    #     'look_yaw_p8d': 4,
    #     'look_yaw_p30d': 5,
    #     'look_pitch_n10d': 6,
    #     'look_pitch_p10d': 7,
    #     'is_grab': 8,
    #     'is_speak': 9
    # }
    action_encoder = {
        'no_op': 0,
        'move_forward': 1,
        'look_yaw_n8d': 2,
        'look_yaw_p8d': 3,
        'look_yaw_p30d': 4,
        'look_pitch_n10d': 5,
        'look_pitch_p10d': 6,
        'is_grab': 7,
        'is_speak': 8
    }
    '''
    if degree_pitch \in [-18,-14-eps]:
        turn left 7
    elif degree_pitch \in [14+eps, ...]:
        turn right 28
    '''
    def get_look_yaw(cur_degree_yaw, tolerance, boundary=20, eps=1):
        assert abs(cur_degree_yaw) > tolerance+eps, f"cur_degree_yaw={cur_degree_yaw}, tolerance={tolerance}"
        if cur_degree_yaw>=-boundary+eps and degree_yaw<-tolerance-eps:
            action_env = {
                "move_right": 0,
                "move_forward": 0,
                "look_yaw": -tolerance*2,
                "look_pitch": 0.0,
                "jump": False,
                "grab": False,
                "speak": "",
                }
            action_str = f"look_yaw_n{int(tolerance*2)}d"
        else: # cur_degree_yaw >= tolerance+eps and cur_degree_yaw<boundary-eps:
            action_env = {
                "move_right": 0,
                "move_forward": 0,
                "look_yaw": tolerance*2,
                "look_pitch": 0.0,
                "jump": False,
                "grab": False,
                "speak": "",
                }
            action_str = f'look_yaw_p{int(tolerance*2)}d'
        return action_env, action_str
    eps = 1
    yaw_boundary = 20
    tiny_yaw_tolerance = 4
    pitch_tolerance = 5
    if abs(degree_yaw) <= yaw_boundary-eps:
        if abs(degree_yaw) > tiny_yaw_tolerance+eps:
            action_env, action_str = get_look_yaw(degree_yaw, tiny_yaw_tolerance)
        elif abs(degree_pitch) <= pitch_tolerance+eps:
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
        elif degree_pitch <-pitch_tolerance-eps:
            action_env = {
                "move_right": 0,
                "move_forward": 0,
                "look_yaw": 0.0,
                "look_pitch": 10,
                "jump": False,
                "grab": False,
                "speak": "",
                }
            action_str = 'look_pitch_p10d'
        else:
            action_env = {
                "move_right": 0,
                "move_forward": 0,
                "look_yaw": 0.0,
                "look_pitch": -10,
                "jump": False,
                "grab": False,
                "speak": "",
                }
            action_str = 'look_pitch_n10d'
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

def collect(env,episodes:int, collect_object:str='player'):
    if collect_object != 'player':
        set_object_counts({collect_object: 1})
        instruction = f"come to the {collect_object.split('_')[0]}"
    else:
        instruction = "come here"
    
    print(f"instruction: come to the {collect_object}, labels_instruction: {instruction}", flush=True)
    env_decoder = ComeHereEnv(goal_object=collect_object,distance_reward=10, success_reward=200, distance_min=1.2,
                               step_penalty=1, episode_len=100, look_yaw_tolerance=15,look_pitch_tolerance=5,eps=0.5, is_debug=True)

    action_noop = {
        "move_right": 0,
        "move_forward": 0,
        "look_yaw": 0.0,
        "look_pitch": 0.0,
        "jump": False,
        "grab": False,
        "speak": "",
        }
    action_grab = {
        "move_right": 0,
        "move_forward": 0,
        "look_yaw": 0.0,
        "look_pitch": 0.0,
        "jump": False,
        "grab": True,
        "speak": "",
    }

    if not os.path.exists('./dataset'):
        os.makedirs('./dataset')

    with h5py.File(f'./dataset/Episode1000_{collect_object}.h5', 'w') as f:
        data_nums = 0
        fail_nums = 0
        success_episode, fail_episode = 0,0
        obs_V_set, obs_T_set, action_set = [],[],[]
        success_grab = 0
        for episode in range(episodes):
            if episode%50==0:
                print(f"Episode: {episode}/{episodes}", flush=True)
            
            state_img, _ = env.reset()
            env_decoder.reset()
            done, blocked = False, False
            (state_img, state_text), _, _, info = env.step(**action_noop)
            obs_list = []

            while not (done or blocked):
                last_state_img = state_img
                last_state_text = state_text
                last_state_text = instruction
                action_env, action_code, action_str = inference_come_to_sth_action(info, goal_object=collect_object)
                (state_img, state_text), _, _, info = env.step(**action_env)
                reward, done, blocked, cumulate_reward, elspsed_step, distance_info = env_decoder.step(info)
                obs_list.append((last_state_img, last_state_text, action_code, f'/{elspsed_step:03d}_{action_str}_{round(reward,1)}.png'))
            obs_list.append((state_img, instruction, 0, f'/{elspsed_step+1:03d}_noOp_NULL.png'))
            if blocked:
                fail_nums += len(obs_list)
                fail_episode += 1
                # print(f"BLOCKED_sum: {fail_nums}")
                continue
            success_episode += 1
            if instruction != "come here":
                (_,_), _,_, info = env.step(**action_grab)
                if not blocked and collect_object in info['game_states']['instances'][info['game_states']['playmateGrabInstance']]['prefab']:
                    success_grab += 1
                else:
                    pass
                    # not_grab_nums += len(obs_list)
                    # continue
            path = f"./obs_visions_{collect_object}/episode_{episode}_{elspsed_step:.0f}_{cumulate_reward:.0f}"
            if not os.path.exists(path):
                os.makedirs(path)

            for state_img,state_text,action_code,pic_path in obs_list:
                data_nums += 1
                obs_V_set.append(state_img)
                obs_T_set.append(state_text)
                action_set.append(action_code)
                save_pic(state_img, pic_path=path+pic_path)

            # os.rename(path, path + f"_{elspsed_step}_{distance_info}_{cumulate_reward}")
        f.create_dataset('obs_V', data=np.array(obs_V_set).astype(np.uint8), compression='gzip', compression_opts=9)
        # f.create_dataset('obs_T', (0,), maxshape=(None,),dtype=h5py.string_dtype("utf-8")) #TODO 往出拿时需要 “ss”.decode("utf-8")
        f.create_dataset('obs_T', data=obs_T_set)
        f.create_dataset('action', data=action_set)
    env.close()
    print(f"Blocked_num: {fail_nums}",flush=True)
    print(f"{success_grab}/{success_episode} success_grab/success_episodes = {success_grab/success_episode}, {fail_episode} fail_episodes, {data_nums} data pairs.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal_object", type=str, default="ChristmasTree_01")
    parser.add_argument("--episodes_num", type=int, default=50)
    args = parser.parse_args()
    # ligent.set_scenes_dir("")
    # env = ligent.Environment(path="C:/Users/19355/Desktop/drlProject/ligent-windows/06061943_win_224/LIGENT.exe")
    # collect(env,50, collect_object='Pumpkin')
    # collect(env,args.episodes_num, collect_object=args.goal_object)
    try:
        ligent.set_scenes_dir("")
        env = ligent.Environment(path="C:/Users/19355/Desktop/drlProject/ligent-windows/06061943_win_224/LIGENT.exe")
        collect(env,args.episodes_num, collect_object=args.goal_object)
    except Exception as e:
        # print(e, flush=True)
        raise
        env.close()