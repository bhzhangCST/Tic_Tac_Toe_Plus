import sys
sys.path.append('..')

import numpy as np
import torch
from game import Game81
from model import PolicyValueNet
from mcts import MCTS
from config import Config


class AIPlayer:
    def __init__(self, model_path: str = None, config: Config = None):
        self.config = config if config else Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PolicyValueNet(self.config).to(self.device)
        
        if model_path:
            self.load_model(model_path)
        
        self.mcts = MCTS(self.model, self.config)
    
    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Loaded model from: {path}")
    
    def get_action(self, game: Game81, temperature: float = 0.0) -> int:
        action_probs = self.mcts.search(game, add_noise=False)
        
        if temperature == 0:
            action = np.argmax(action_probs)
        else:
            action_probs = np.power(action_probs, 1 / temperature)
            action_probs = action_probs / action_probs.sum()
            action = np.random.choice(81, p=action_probs)
        
        return action
    
    def get_action_with_probs(self, game: Game81) -> tuple:
        action_probs = self.mcts.search(game, add_noise=False)
        action = np.argmax(action_probs)
        return action, action_probs


class RandomPlayer:
    def get_action(self, game: Game81, temperature: float = 0.0) -> int:
        valid_moves = game.get_valid_moves()
        return np.random.choice(valid_moves)
