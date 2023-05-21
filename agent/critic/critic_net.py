import torch.nn as nn
from ..utils import build_mlp


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, hidden_depth, device, activation=nn.ELU, critic_dropout=0.0):
        super().__init__()
        # super().__init__(num_states, 0, hidden_dims, output_activation=nn.Identity,
        #                  activation=activation, critic_dropout=critic_dropout)
        self.mlp =  build_mlp(
                input_dim=state_dim,
                output_dim=1,
                hidden_dim=hidden_dim,
                hidden_depth=hidden_depth,
                activation=activation,
                norm_type=None,).to(device)
        
    def forward(self, state):
        return self.mlp(state).squeeze()