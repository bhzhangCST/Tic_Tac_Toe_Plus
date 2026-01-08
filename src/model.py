import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class PolicyValueNet(nn.Module):
    def __init__(self, config: Config = None):
        super().__init__()
        if config is None:
            config = Config()
        
        self.input_conv = nn.Sequential(
            nn.Conv2d(6, config.num_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(config.num_channels),
            nn.ReLU()
        )
        
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(config.num_channels) for _ in range(config.num_res_blocks)]
        )
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(config.num_channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 81)
        )
        
        self.value_head = nn.Sequential(
            nn.Conv2d(config.num_channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.input_conv(x)
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
    
    def predict(self, state, valid_moves):
        self.eval()
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            
            if len(state.shape) == 3:
                state = state.unsqueeze(0)
            
            if next(self.parameters()).is_cuda:
                state = state.cuda()
            
            policy, value = self(state)
            policy = F.softmax(policy, dim=1).cpu().numpy()[0]
            
            mask = np.zeros(81)
            mask[valid_moves] = 1
            policy = policy * mask
            policy_sum = policy.sum()
            if policy_sum > 0:
                policy = policy / policy_sum
            else:
                policy[valid_moves] = 1.0 / len(valid_moves)
            
            return policy, value.cpu().numpy()[0][0]
