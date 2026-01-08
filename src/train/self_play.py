import sys
sys.path.append('..')

import numpy as np
from typing import List, Tuple
from game import Game81
from mcts import MCTS
from config import Config


def execute_episode(model, config: Config) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    game = Game81()
    mcts = MCTS(model, config)
    episode_data = []
    
    while not game.is_terminal():
        state = game.get_canonical_state()
        action_probs = mcts.search(game, add_noise=True)
        
        episode_data.append((state, action_probs, game.current_player))
        
        if game.move_count < config.temperature_threshold:
            action = np.random.choice(81, p=action_probs)
        else:
            action = np.argmax(action_probs)
        
        game.make_move(action)
    
    winner = game.check_winner()
    training_data = []
    
    for state, action_probs, player in episode_data:
        if winner == 0:
            value = 0.0
        elif winner == player:
            value = 1.0
        else:
            value = -1.0
        training_data.append((state, action_probs, value))
    
    return training_data


def augment_data(data: List[Tuple[np.ndarray, np.ndarray, float]]) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    augmented = []
    for state, policy, value in data:
        policy_2d = policy.reshape(9, 9)
        
        for k in range(4):
            rot_state = np.rot90(state, k, axes=(1, 2))
            rot_policy = np.rot90(policy_2d, k).flatten()
            augmented.append((rot_state, rot_policy, value))
            
            flip_state = np.flip(rot_state, axis=2)
            flip_policy = np.fliplr(np.rot90(policy_2d, k)).flatten()
            augmented.append((flip_state, flip_policy, value))
    
    return augmented
