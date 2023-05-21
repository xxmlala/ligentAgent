import torch
import torch.nn as nn
# from ..batch import Batch
from ..utils import build_mlp
# from mineclip.utils import call_once

class SimpleCNNEncoder(nn.Module):
    def __init__(self, input_size, output_size, device):
        super().__init__()
        input_H, input_W, num_channel = input_size
        assert input_H % 8 == 0 and input_W % 8 == 0
        self.device = device
        self.input_H, self.input_W = input_H, input_W
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * self.input_H//8 * self.input_W//8, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = torch.as_tensor(x, device=self.device).to(torch.float32)
        x = x.transpose(-3,-1)
        x /= 255
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.pool(x)
        x = x.view(-1, 128 * self.input_H//8 * self.input_W//8)
        x = self.fc1(x)
        return x


class DummyTextEncoder(nn.Module):
    def __init__(self, input_size, output_size:int, device):
        super().__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.device = device
        self.linear = nn.Linear(self.input_size,self.output_size)
    def forward(self, x):  
        # if x is not torch.tensor:
        x = torch.as_tensor(x, device=self.device)
        if x.ndim<2:
            x = x.unsqueeze(0)
        return self.linear(x) * 0
        # return torch.zeros(self.output_size)


class FeatureFusion(nn.Module):
    def __init__(self, hidden_depth, text_input_size:int, text_hidden_size:int, img_input_size:list[int], img_hidden_size:int, 
                   device, text_encoder:nn.Module=DummyTextEncoder, img_encoder:nn.Module=SimpleCNNEncoder):
        super().__init__()
        self.text_encoder = text_encoder(text_input_size, text_hidden_size, device).to(device)
        self.img_encoder = img_encoder(img_input_size, img_hidden_size, device).to(device)
        self.text_output_size = text_hidden_size
        self._output_dim = img_hidden_size + text_hidden_size
        self._head = build_mlp(
            input_dim=self._output_dim,
            hidden_dim=self._output_dim,
            output_dim=self._output_dim,
            hidden_depth=hidden_depth,
            activation="relu",
            weight_init="orthogonal",
            bias_init="zeros",
            norm_type=None,
            # add input activation because we assume upstream extractors do not have activation at the end
            add_input_activation=True,
            add_input_norm=False,
            add_output_activation=True,
            add_output_norm=False,
        ).to(device)
    @property
    def output_dim(self):
        return self._output_dim
    
    def forward(self, obs_img, obs_text):
        state_img = self.img_encoder(obs_img)
        state_text = self.text_encoder(obs_text)
        # if self.text_encoder is None:
        #     if len(state_img.shape)==2:
        #         state_text = torch.zeros((len(state_img), self.text_output_size))
        #     else:
        #         state_text = torch.zeros(self.text_output_size)
        return self._head(torch.concat([state_img, state_text], dim=-1))