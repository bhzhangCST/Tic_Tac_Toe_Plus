import sys
sys.path.append('..')

import argparse
from config import Config
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train AlphaZero for 81-cell chess')
    parser.add_argument('--iterations', type=int, default=None)
    parser.add_argument('--games-per-iter', type=int, default=None)
    parser.add_argument('--epochs-per-iter', type=int, default=None)
    parser.add_argument('--mcts-sims', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    config = Config()
    
    if args.iterations:
        config.num_iterations = args.iterations
    if args.games_per_iter:
        config.games_per_iteration = args.games_per_iter
    if args.epochs_per_iter:
        config.epochs_per_iteration = args.epochs_per_iter
    if args.mcts_sims:
        config.num_simulations = args.mcts_sims
    
    print("Training configuration:")
    print(f"  Iterations: {config.num_iterations}")
    print(f"  Games per iteration: {config.games_per_iteration}")
    print(f"  Epochs per iteration: {config.epochs_per_iteration}")
    print(f"  MCTS simulations: {config.num_simulations}")
    
    trainer = Trainer(config)
    
    if args.resume:
        print(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    trainer.train(
        num_iterations=config.num_iterations,
        games_per_iter=config.games_per_iteration,
        epochs_per_iter=config.epochs_per_iteration
    )


if __name__ == "__main__":
    main()
