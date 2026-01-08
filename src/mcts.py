import numpy as np
import math
from typing import Dict, List, Optional
from game import Game81
from config import Config


class MCTSNode:
    def __init__(self, prior: float, parent: Optional['MCTSNode'] = None):
        self.prior = prior
        self.parent = parent
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visit_count = 0
        self.value_sum = 0.0
    
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, c_puct: float) -> float:
        prior_score = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return self.q_value() + prior_score
    
    def is_expanded(self) -> bool:
        return len(self.children) > 0


class MCTS:
    def __init__(self, model, config: Config = None):
        self.model = model
        self.config = config if config else Config()
    
    def search(self, game: Game81, add_noise: bool = True) -> np.ndarray:
        root = MCTSNode(prior=0.0)
        
        valid_moves = game.get_valid_moves()
        state = game.get_canonical_state()
        policy, value = self.model.predict(state, valid_moves)
        
        if add_noise:
            noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(valid_moves))
            for i, move in enumerate(valid_moves):
                noisy_prior = (1 - self.config.dirichlet_epsilon) * policy[move] + \
                              self.config.dirichlet_epsilon * noise[i]
                root.children[move] = MCTSNode(prior=noisy_prior, parent=root)
        else:
            for move in valid_moves:
                root.children[move] = MCTSNode(prior=policy[move], parent=root)
        
        root.visit_count = 1
        
        for _ in range(self.config.num_simulations):
            node = root
            sim_game = game.copy()
            path = [node]
            
            while node.is_expanded() and not sim_game.is_terminal():
                action, node = self._select_child(node)
                sim_game.make_move(action)
                path.append(node)
            
            if sim_game.is_terminal():
                winner = sim_game.check_winner()
                if winner == 0:
                    value = 0.0
                elif winner == sim_game.current_player:
                    value = 1.0
                else:
                    value = -1.0
            else:
                valid_moves = sim_game.get_valid_moves()
                if len(valid_moves) > 0:
                    state = sim_game.get_canonical_state()
                    policy, value = self.model.predict(state, valid_moves)
                    for move in valid_moves:
                        node.children[move] = MCTSNode(prior=policy[move], parent=node)
            
            self._backpropagate(path, value, sim_game.current_player, game.current_player)
        
        action_probs = np.zeros(81)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count
        
        if action_probs.sum() > 0:
            action_probs = action_probs / action_probs.sum()
        
        return action_probs
    
    def _select_child(self, node: MCTSNode):
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in node.children.items():
            score = child.ucb_score(self.config.c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def _backpropagate(self, path: List[MCTSNode], value: float, 
                       current_player: int, root_player: int):
        for node in reversed(path):
            node.visit_count += 1
            if current_player == root_player:
                node.value_sum += value
            else:
                node.value_sum -= value
            current_player = 3 - current_player
