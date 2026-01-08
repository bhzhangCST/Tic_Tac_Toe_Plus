import sys
sys.path.append('..')

import argparse
from tqdm import tqdm
from game import Game81
from player import AIPlayer, RandomPlayer
from config import Config


def evaluate_vs_random(ai_player: AIPlayer, num_games: int = 100) -> dict:
    random_player = RandomPlayer()
    results = {'ai_wins': 0, 'random_wins': 0, 'draws': 0}
    
    for i in tqdm(range(num_games), desc="Evaluating vs Random"):
        game = Game81()
        ai_is_player1 = (i % 2 == 0)
        
        while not game.is_terminal():
            if (game.current_player == 1) == ai_is_player1:
                action = ai_player.get_action(game, temperature=0.0)
            else:
                action = random_player.get_action(game)
            game.make_move(action)
        
        winner = game.check_winner()
        if winner == 0:
            results['draws'] += 1
        elif (winner == 1) == ai_is_player1:
            results['ai_wins'] += 1
        else:
            results['random_wins'] += 1
    
    return results


def evaluate_self_play(ai_player: AIPlayer, num_games: int = 50) -> dict:
    results = {'player1_wins': 0, 'player2_wins': 0, 'draws': 0}
    
    for _ in tqdm(range(num_games), desc="Self-play evaluation"):
        game = Game81()
        
        while not game.is_terminal():
            action = ai_player.get_action(game, temperature=0.1)
            game.make_move(action)
        
        winner = game.check_winner()
        if winner == 0:
            results['draws'] += 1
        elif winner == 1:
            results['player1_wins'] += 1
        else:
            results['player2_wins'] += 1
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate AI model')
    parser.add_argument('--model', type=str, default='../../checkpoints/latest.pt')
    parser.add_argument('--num-games', type=int, default=100)
    parser.add_argument('--mcts-sims', type=int, default=400)
    parser.add_argument('--mode', choices=['random', 'self'], default='random')
    args = parser.parse_args()
    
    config = Config()
    config.num_simulations = args.mcts_sims
    
    try:
        ai_player = AIPlayer(args.model, config)
    except FileNotFoundError:
        print(f"模型文件未找到: {args.model}")
        return
    
    if args.mode == 'random':
        results = evaluate_vs_random(ai_player, args.num_games)
        print(f"\n=== AI vs Random Player ===")
        print(f"AI胜率: {results['ai_wins'] / args.num_games * 100:.1f}%")
        print(f"Random胜率: {results['random_wins'] / args.num_games * 100:.1f}%")
        print(f"平局率: {results['draws'] / args.num_games * 100:.1f}%")
    else:
        results = evaluate_self_play(ai_player, args.num_games)
        print(f"\n=== Self-Play Evaluation ===")
        print(f"先手胜率: {results['player1_wins'] / args.num_games * 100:.1f}%")
        print(f"后手胜率: {results['player2_wins'] / args.num_games * 100:.1f}%")
        print(f"平局率: {results['draws'] / args.num_games * 100:.1f}%")


if __name__ == "__main__":
    main()
