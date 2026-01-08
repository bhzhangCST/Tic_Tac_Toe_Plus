import sys
sys.path.append('..')

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple
from tqdm import tqdm

from game import Game81
from model import PolicyValueNet
from mcts import MCTS
from config import Config
from self_play import execute_episode, augment_data


class Trainer:
    def __init__(self, config: Config = None):
        self.config = config if config else Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PolicyValueNet(self.config).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.replay_buffer: List[Tuple[np.ndarray, np.ndarray, float]] = []
    
    def self_play(self, num_games: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        self.model.eval()
        all_data = []
        
        for _ in tqdm(range(num_games), desc="Self-play"):
            episode_data = execute_episode(self.model, self.config)
            augmented = augment_data(episode_data)
            all_data.extend(augmented)
        
        return all_data
    
    def train_epoch(self, batch_size: int) -> Tuple[float, float, float]:
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0, 0.0
        
        self.model.train()
        batch = random.sample(self.replay_buffer, min(len(self.replay_buffer), batch_size * 10))
        
        states = torch.FloatTensor(np.array([x[0] for x in batch])).to(self.device)
        target_policies = torch.FloatTensor(np.array([x[1] for x in batch])).to(self.device)
        target_values = torch.FloatTensor(np.array([x[2] for x in batch])).to(self.device).unsqueeze(1)
        
        dataset = TensorDataset(states, target_policies, target_values)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        num_batches = 0
        
        for state_batch, policy_batch, value_batch in loader:
            self.optimizer.zero_grad()
            
            policy_pred, value_pred = self.model(state_batch)
            
            policy_loss = -torch.mean(torch.sum(policy_batch * torch.log_softmax(policy_pred, dim=1), dim=1))
            value_loss = nn.MSELoss()(value_pred, value_batch)
            loss = policy_loss + value_loss
            
            loss.backward()
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches, total_policy_loss / num_batches, total_value_loss / num_batches
    
    def train(self, num_iterations: int, games_per_iter: int, epochs_per_iter: int):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        for iteration in range(1, num_iterations + 1):
            print(f"\n=== Iteration {iteration}/{num_iterations} ===")
            
            new_data = self.self_play(games_per_iter)
            self.replay_buffer.extend(new_data)
            
            if len(self.replay_buffer) > self.config.replay_buffer_size:
                self.replay_buffer = self.replay_buffer[-self.config.replay_buffer_size:]
            
            print(f"Replay buffer size: {len(self.replay_buffer)}")
            
            for epoch in range(epochs_per_iter):
                loss, p_loss, v_loss = self.train_epoch(self.config.batch_size)
                print(f"Epoch {epoch+1}: loss={loss:.4f}, policy={p_loss:.4f}, value={v_loss:.4f}")
            
            if iteration % 5 == 0 or iteration == num_iterations:
                self.save_checkpoint(iteration)
    
    def save_checkpoint(self, iteration: int):
        path = os.path.join(self.config.checkpoint_dir, f"model_iter_{iteration}.pt")
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        
        latest_path = os.path.join(self.config.checkpoint_dir, "latest.pt")
        torch.save({
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, latest_path)
        
        print(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint.get('iteration', 0)
