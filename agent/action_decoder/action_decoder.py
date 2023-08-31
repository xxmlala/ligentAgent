import torch.nn as nn
import torch

class ActionDecoder:
    def __init__(self, decode_idx, action_used_dims, action_env_dims, device):
        self.decode_idx = decode_idx
        self.action_used_dims = action_used_dims
        self.action_env_dims = action_env_dims
        self.device = device

    def decode(self, action_used):
        action_code = torch.as_tensor(action_used, device=self.device) #[, 2] (foward; rotate 30 degree to right)
        # eqa_strs = torch.zeros(list(action_used.shape[:-1])+[self.action_env_dims-self.action_used_dims+1], device=self.device)
        # action_env = torch.cat([action_used[...,:-1], eqa_strs], dim=-1)
        action_code = action_code.squeeze(0).detach().cpu().numpy()
        
        # 3 classes
        # if action_used[0] == 0:
        #     action_env = { 
        #         "move_right": 0,
        #         "move_forward": 1,
        #         "look_yaw": 0,
        #         "look_pitch": 0,
        #         "jump": False,
        #         "grab": False,
        #         "speak": "",
        #     }
        # elif action_used[0] == 1:
        #     action_env = { 
        #         "move_right": 0,
        #         "move_forward": 0,
        #         "look_yaw": -30,
        #         "look_pitch": 0,
        #         "jump": False,
        #         "grab": False,
        #         "speak": "",
        #     }
        # elif action_used[0] == 2:
        #     action_env = { 
        #         "move_right": 0,
        #         "move_forward": 0,
        #         "look_yaw": 30,
        #         "look_pitch": 0,
        #         "jump": False,
        #         "grab": False,
        #         "speak": "",
        #     }
        # else:
        #     raise Exception

        # if action_used[0] == 0:
        #     action_env = { 
        #         "move_right": 0,
        #         "move_forward": 0,
        #         "look_yaw": 0,
        #         "look_pitch": 0,
        #         "jump": False,
        #         "grab": False,
        #         "speak": "",
        #     }
        # elif action_used[0] == 1:
        #     action_env = { 
        #         "move_right": 0,
        #         "move_forward": 1,
        #         "look_yaw": 0,
        #         "look_pitch": 0,
        #         "jump": False,
        #         "grab": False,
        #         "speak": "",
        #     }
        # elif action_used[0] == 2:
        #     action_env = { 
        #         "move_right": 0,
        #         "move_forward": 0,
        #         "look_yaw": -30,
        #         "look_pitch": 0,
        #         "jump": False,
        #         "grab": False,
        #         "speak": "",
        #     }
        # elif action_used[0] == 3:
        #     action_env = { 
        #         "move_right": 0,
        #         "move_forward": 0,
        #         "look_yaw": -8,
        #         "look_pitch": 0,
        #         "jump": False,
        #         "grab": False,
        #         "speak": "",
        #     }
        # elif action_used[0] == 4:
        #     action_env = { 
        #         "move_right": 0,
        #         "move_forward": 0,
        #         "look_yaw": 8,
        #         "look_pitch": 0,
        #         "jump": False,
        #         "grab": False,
        #         "speak": "",
        #     }
        # elif action_used[0] == 5:
        #     action_env = { 
        #         "move_right": 0,
        #         "move_forward": 0,
        #         "look_yaw": 30,
        #         "look_pitch": 0,
        #         "jump": False,
        #         "grab": False,
        #         "speak": "",
        #     }
        # elif action_used[0] == 6:
        #     action_env = { 
        #         "move_right": 0,
        #         "move_forward": 0,
        #         "look_yaw": 0,
        #         "look_pitch": -10,
        #         "jump": False,
        #         "grab": False,
        #         "speak": "",
        #     }
        # elif action_used[0] == 7:
        #     action_env = { 
        #         "move_right": 0,
        #         "move_forward": 0,
        #         "look_yaw": 0,
        #         "look_pitch": 10,
        #         "jump": False,
        #         "grab": False,
        #         "speak": "",
        #     }
        # else:
        #     raise Exception


        if action_used[0] == 0:
            action_env = { 
                "move_right": 0,
                "move_forward": 0,
                "look_yaw": 0,
                "look_pitch": 0,
                "jump": False,
                "grab": False,
                "speak": "",
            }
        elif action_used[0] == 1:
            action_env = { 
                "move_right": 0,
                "move_forward": 1,
                "look_yaw": 0,
                "look_pitch": 0,
                "jump": False,
                "grab": False,
                "speak": "",
            }
        elif action_used[0] == 2:
            action_env = { 
                "move_right": 0,
                "move_forward": 0,
                "look_yaw": -8,
                "look_pitch": 0,
                "jump": False,
                "grab": False,
                "speak": "",
            }
        elif action_used[0] == 3:
            action_env = { 
                "move_right": 0,
                "move_forward": 0,
                "look_yaw": 8,
                "look_pitch": 0,
                "jump": False,
                "grab": False,
                "speak": "",
            }
        elif action_used[0] == 4:
            action_env = { 
                "move_right": 0,
                "move_forward": 0,
                "look_yaw": 30,
                "look_pitch": 0,
                "jump": False,
                "grab": False,
                "speak": "",
            }
        elif action_used[0] == 5:
            action_env = { 
                "move_right": 0,
                "move_forward": 0,
                "look_yaw": 0,
                "look_pitch": -10,
                "jump": False,
                "grab": False,
                "speak": "",
            }
        elif action_used[0] == 6:
            action_env = { 
                "move_right": 0,
                "move_forward": 0,
                "look_yaw": 0,
                "look_pitch": 10,
                "jump": False,
                "grab": False,
                "speak": "",
            }
        else:
            raise Exception
        return action_env