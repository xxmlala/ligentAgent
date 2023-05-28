import torch
import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self, feature_net, actor_net):
        super().__init__()
        self.feature_net = feature_net
        self.actor_net = actor_net
        
    def forward(self, obs_V, obs_T):
        obs = self.feature_net(obs_V, obs_T)
        action = self.actor_net.forward_logits(obs)
        return action