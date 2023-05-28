import torch
import torch.nn as nn

from ..utils import build_mlp
from .distribution import MultiCategorical


class MultiCategoricalActor(nn.Module):
    def __init__(
        self,
        # preprocess_net: nn.Module,
        state_dim:int,
        # *,
        action_dim: list[int],
        hidden_dim: int,
        hidden_depth: int,
        device,
        activation: str = "relu",
    ):
        super().__init__()
        self.mlps = nn.ModuleList()
        # self.preprocess = preprocess_net
        for action in action_dim:
            net = build_mlp(
                # input_dim=preprocess_net.output_dim,
                input_dim=state_dim,
                output_dim=action,
                hidden_dim=hidden_dim,
                hidden_depth=hidden_depth,
                activation=activation,
                norm_type=None,
                add_input_activation=True #we assume feature_net's output is not activated
            ).to(device)
            self.mlps.append(net)
        self._action_dim = action_dim
        self._device = device

    def forward_logits(self, state):
        hidden = None
        # x, _ = self.preprocess(x)
        return torch.cat([mlp(state) for mlp in self.mlps], dim=1)

    @property
    def dist_fn(self):
        return lambda x: MultiCategorical(logits=x, action_dims=self._action_dim)

    def evaluate_actions(self, state, action) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward_logits(state)
        dist = self.dist_fn(logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        #TODO check the shape of log_prob and extropy
        return log_prob, entropy
    
    def forward(self, state, sample=True):
        logits = self.forward_logits(state)
        dist = self.dist_fn(logits)
        if not sample:
            return dist.mode(), None
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob       