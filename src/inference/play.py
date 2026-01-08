import sys
sys.path.append('..')

import argparse
from game import Game81
from player import AIPlayer
from config import Config


def human_input(game: Game81) -> int:
    valid_moves = game.get_valid_moves()
    print(f"\n可落子位置: ", end="")
    for i, move in enumerate(valid_moves):
        print(f"{game.action_to_string(move)}", end=" ")
        if (i + 1) % 9 == 0:
            print()
    
    while True:
        try:
            user_input = input("\n请输入落子位置 (行 列，如 4 5): ").strip()
            if user_input.lower() == 'q':
                return -1
            
            parts = user_input.split()
            if len(parts) == 2:
                row, col = int(parts[0]), int(parts[1])
                action = row * 9 + col
                if action in valid_moves:
                    return action
                else:
                    print("无效位置，请重新输入")
            else:
                print("格式错误，请输入: 行 列")
        except ValueError:
            print("输入错误，请重新输入")


def play_game(ai_player: AIPlayer, human_first: bool = True):
    game = Game81()
    human_symbol = 1 if human_first else 2
    ai_symbol = 2 if human_first else 1
    
    print("\n=== 八十一子棋 人机对战 ===")
    print(f"你是: {'X (先手)' if human_first else 'O (后手)'}")
    print("输入 'q' 退出游戏\n")
    
    while not game.is_terminal():
        game.display()
        
        if game.current_player == human_symbol:
            action = human_input(game)
            if action == -1:
                print("游戏结束")
                return
        else:
            print("\nAI思考中...")
            action = ai_player.get_action(game, temperature=0.0)
            print(f"AI落子: {game.action_to_string(action)}")
        
        game.make_move(action)
    
    game.display()
    winner = game.check_winner()
    
    if winner == human_symbol:
        print("\n恭喜你获胜!")
    elif winner == ai_symbol:
        print("\nAI获胜!")
    else:
        print("\n平局!")


def main():
    parser = argparse.ArgumentParser(description='Play against AI')
    parser.add_argument('--model', type=str, default='../../checkpoints/latest.pt')
    parser.add_argument('--human-first', action='store_true', default=True)
    parser.add_argument('--mcts-sims', type=int, default=800)
    args = parser.parse_args()
    
    config = Config()
    config.num_simulations = args.mcts_sims
    
    try:
        ai_player = AIPlayer(args.model, config)
    except FileNotFoundError:
        print(f"模型文件未找到: {args.model}")
        print("请先运行训练程序生成模型")
        return
    
    while True:
        play_game(ai_player, human_first=args.human_first)
        
        again = input("\n再来一局? (y/n): ").strip().lower()
        if again != 'y':
            break
    
    print("感谢游玩!")


if __name__ == "__main__":
    main()
