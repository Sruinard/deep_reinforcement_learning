

class Config:
    n_envs = 32
    env_seed = 42

    n_epochs = 100000
    n_updates_per_rollout = 4
    horizon = 256
    mini_batch_size = 4

    learning_rate = 0.003
    model_seed = 42

    gamma = 0.99
    gae_lambda = 0.95
    clip_eps = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.01
