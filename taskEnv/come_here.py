import numpy as np
import math
class ComeHereEnv:
    def __init__(
        self,
        distance_reward: float | int,
        success_reward: float | int,
        distance_min: float | int,
        step_penalty: float | int,
        episode_len,
        look_yaw_tolerance,
        look_pitch_tolerance,
        eps,
        goal_object: str='',
        is_debug=False
    ):
        self._goal_object = goal_object
        self._episode_len = episode_len
        self._distance_reward = distance_reward
        self._success_reward = success_reward
        self._distance_min = distance_min
        self._step_penalty = step_penalty

        self._elapsed_step = 0
        self._cumulate_reward = 0
        self._gotten_min_distance = -1
        self._init_distance = -1
        
        self._stay_static_steps = 0
        self._last_distance = 9999

        self._look_yaw_tolerance = look_yaw_tolerance
        self._look_pitch_tolerance = look_pitch_tolerance
        self._eps = eps

        self._is_debug = is_debug

    def reset(self):
        _ret = self._elapsed_step
        self._elapsed_step = 0
        self._cumulate_reward = 0
        self._gotten_min_distance = -1
        self._init_distance = -1
        self._stay_static_steps = 0
        self._last_distance = 9999

    def step(self, info):
        self._elapsed_step += 1
        position_mate = info['game_states']['playmate']['position']
        position_camera_mate = info['game_states']['playmateCamera']['position']
        face_direction_camera_mate = info['game_states']['playmateCamera']['forward']
        face_direction_mate = info['game_states']['playmate']['forward']

        if self._goal_object=='player':
            position_goal = info['game_states']['player']['position']
            position_camera_player = info['game_states']['playerCamera']['position']
            x2z_norm = ((position_camera_player['x']-position_camera_mate['x'])**2 + \
                        (position_camera_player['z']-position_camera_mate['z'])**2)**0.5
            camera_mate2goal = np.array([position_camera_player['y']-position_camera_mate['y'], x2z_norm])
        else:
            instances_list = [i['prefab'] for i in info['game_states']['instances']]
            goal_object_ids = []
            for i,instance in enumerate(instances_list):
                # if instance==self._goal_object:
                if self._goal_object in instance:
                    goal_object_ids.append(i)
            assert len(goal_object_ids)==1,f"len(collect_object_idx) is {len(goal_object_ids)}, goal={self._goal_object}"
            goal_object_id = goal_object_ids[0]
            position_goal = info['game_states']['instances'][goal_object_id]['position']
            x2z_norm = ((position_goal['x']-position_camera_mate['x'])**2 + \
                    (position_goal['z']-position_camera_mate['z'])**2)**0.5
            camera_mate2goal = np.array([position_goal['y']-position_camera_mate['y'], x2z_norm])
    
        # face_direction_player = info['game_states']['player']['forward']
        # face_direction_mate = info['game_states']['playmate']['forward']
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
        
        def get_distance(player:dict, mate:dict):
            distance = ((player['x']-mate['x'])**2 + (player['z']-mate['z'])**2) ** 0.5
            return distance
        mate2goal = np.array([position_goal['x']-position_mate['x'], position_goal['z']-position_mate['z']])
        face_direction_mate = np.array([face_direction_mate['x'], face_direction_mate['z']])
        x2z_norm = (face_direction_camera_mate['x']**2 + face_direction_camera_mate['z']**2)**0.5
        face_direction_camera_mate = np.array([face_direction_camera_mate['y'],x2z_norm])   
        degree_yaw = deg(mate2goal, face_direction_mate)
        degree_pitch = deg(camera_mate2goal, face_direction_camera_mate)

        distance_current = get_distance(position_goal, position_mate)
        if self._gotten_min_distance == -1:
            self._init_distance = distance_current
            self._gotten_min_distance = distance_current

        if abs(distance_current-self._last_distance) < 0.1:
            self._stay_static_steps += 1
        else:
            self._stay_static_steps = 0
        blocked = self._stay_static_steps >= 15
        self._last_distance = distance_current

        d_min_t = min(self._gotten_min_distance, distance_current)

        reward = self._distance_reward * max(self._gotten_min_distance-d_min_t, 0) - self._step_penalty
        done = distance_current <= self._distance_min and \
                abs(degree_pitch)<=self._look_pitch_tolerance+self._eps and \
                abs(degree_yaw)<=self._look_yaw_tolerance+self._eps
        
        blocked = blocked or self._elapsed_step >= self._episode_len
        if distance_current <= self._distance_min:
            reward += self._success_reward
        self._cumulate_reward += reward
        self._gotten_min_distance = d_min_t
        if not self._is_debug:
            return reward, done, blocked, self._cumulate_reward, self._elapsed_step
        else:
            return reward, done, blocked, self._cumulate_reward, self._elapsed_step, \
                    (self._init_distance, self._gotten_min_distance, distance_current) #(init d, d_min, final_d)