class ComeHereEnv:
    def __init__(
        self,
        distance_reward: float | int,
        success_reward: float | int,
        distance_min: float | int,
        step_penalty: float | int,
        episode_len,
        is_debug=False
    ):
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

        position_player = info['game_states']['player']['position']
        position_mate = info['game_states']['playmate']['position']

        face_direction_player = info['game_states']['player']['forward']
        face_direction_mate = info['game_states']['playmate']['forward']

        def get_distance(player:dict, mate:dict):
            distance = ((player['x']-mate['x'])**2 + (player['z']-mate['z'])**2) ** 0.5
            return distance
        
        distance_current = get_distance(position_player, position_mate)
        if self._gotten_min_distance == -1:
            self._init_distance = distance_current
            self._gotten_min_distance = distance_current


        if abs(distance_current-self._last_distance) < 0.1:
            self._stay_static_steps += 1
        else:
            self._stay_static_steps = 0
        blocked = self._stay_static_steps >= 10
        self._last_distance = distance_current

        d_min_t = min(self._gotten_min_distance, distance_current)

        reward = self._distance_reward * max(self._gotten_min_distance-d_min_t, 0) - self._step_penalty
        done = distance_current <= self._distance_min or self._elapsed_step >= self._episode_len
        if distance_current <= self._distance_min:
            reward += self._success_reward
        self._cumulate_reward += reward
        self._gotten_min_distance = d_min_t
        if not self._is_debug:
            return reward, done, blocked, self._cumulate_reward, self._elapsed_step
        else:
            return reward, done, blocked, self._cumulate_reward, self._elapsed_step, \
                    (self._init_distance, self._gotten_min_distance, distance_current) #(init d, d_min, final_d)