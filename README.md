# Tic_Tac_Toe_Plus

基于 AlphaZero 的八十一子棋 AI 训练与推理框架。

## Dependencies

```bash
pip install -r requirements.txt
```

## Inference

```bash
cd src/inference

# Human vs AI
python play.py --model ../../checkpoints/latest.pt
# AI vs Random Player
python evaluate.py --model ../../checkpoints/latest.pt --num-games 100
# Self Play
python evaluate.py --model ../../checkpoints/latest.pt --mode self --num-games 50
```

## Train

```bash
cd src/train
# Train
python run_train.py --iterations 200 --games-per-iter 100 --mcts-sims 800
# Resume
python run_train.py --iterations 100 --resume ../../checkpoints/latest.pt
```

