"""
Env 2D
@author: huiming zhou
"""


class Env:
    def __init__(self, _map=None):
        # self.x_range = 51  # size of background
        # self.y_range = 31
        self.x_range, self.y_range = _map.shape[0], _map.shape[1]
        self.motions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                        (1, 0), (1, -1), (0, -1), (-1, -1)]
        if _map is None:
            self.obs = self.obs_map()
        else:
            self.obs = self.preprocess_obs_map(_map)

    def update_obs(self, obs):
        self.obs = obs

    def obs_map(self):
        """
        Initialize obstacles' positions
        :return: map of obstacles
        """

        x = self.x_range
        y = self.y_range
        obs = set()

        for i in range(x):
            obs.add((i, 0))
        for i in range(x):
            obs.add((i, y - 1))

        for i in range(y):
            obs.add((0, i))
        for i in range(y):
            obs.add((x - 1, i))

        for i in range(10, 21):
            obs.add((i, 15))
        for i in range(15):
            obs.add((20, i))

        for i in range(15, 30):
            obs.add((30, i))
        for i in range(16):
            obs.add((40, i))

        return obs

    def preprocess_obs_map(self, _map):
        nonzero_idxs = _map.nonzero()
        coordinate_list = set(list(zip(nonzero_idxs[0], nonzero_idxs[1])))
        return coordinate_list 
        # raise NotImplementedError
    
    def get_obs_map(self):
        return self.obs