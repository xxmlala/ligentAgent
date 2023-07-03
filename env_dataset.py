from torch.utils.data import Dataset, DataLoader

class EnvDataset(Dataset):
    def __init__(self, obs_V, obs_T, action) -> None:
        super().__init__()
        self.obs_V = obs_V
        self.obs_T = obs_T
        self.action = action

    def __len__(self):
        return len(self.action)        
    
    def __getitem__(self, index):
        return self.obs_V[index], self.obs_T[index].decode('utf-8'), self.action[index]