class Config:
    # ========== 基础参数 ==========
    board_size = 9
    big_board_size = 3
    
    # ========== 网络结构参数 ==========
    num_channels = 128
    num_res_blocks = 5
    
    # ========== MCTS 参数 ==========
    num_simulations = 800
    c_puct = 1.5
    dirichlet_alpha = 0.3
    dirichlet_epsilon = 0.25
    
    # ========== 训练参数 ==========
    num_iterations = 100
    games_per_iteration = 100
    epochs_per_iteration = 10
    batch_size = 256
    learning_rate = 0.001
    weight_decay = 1e-4
    replay_buffer_size = 50000
    temperature = 1.0
    temperature_threshold = 30
    
    # ========== 推理参数 ==========
    inference_mcts_sims = 400
    inference_temperature = 0.0
    
    # ========== 路径配置 ==========
    checkpoint_dir = "../../checkpoints"

