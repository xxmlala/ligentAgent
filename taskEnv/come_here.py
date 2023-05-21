class ComeHereEnv:
    def __init__(
        self,
        distance_reward: float | int,
        distance_min: float | int,
        episode_len
    ):
        self._episode_len = episode_len
        self._distance_reward = distance_reward
        self._distance_min = distance_min
        self._elapsed_step = 0
        self._cumulate_reward = 0

    def reset(self):
        self._elapsed_step = 0
        self._cumulate_reward = 0

    def step(self, info):
        self._elapsed_steps += 1

        position_player = info['player']['position']
        position_mate = info['playmate']['position']

        face_direction_player = info['player']['forward']
        face_direction_mate = info['playmate']['forward']

        def distanceR(player:dict, mate:dict):
            distance = ((player['x']-mate['x'])**2 + (player['z']-mate['z'])**2) ** 0.5
            return 1/(1e-2+distance), distance
        
        distance_coef, distance = distanceR(position_player, position_mate)
        reward = distance_coef * self._distance_reward
        done = distance < self._distance_min or self._elapsed_steps >= self._episode_len
        self._cumulate_reward += reward
        return reward, done, self._cumulate_reward