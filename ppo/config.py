

class Config:
    n_envs = 10
    env_seed = 0

    n_epochs = 100
    n_updates_per_rollout = 4
    horizon = 12

    learning_rate = 1e-3
    model_seed = 42

    gamma = 0.99
    gae_lambda = 0.95
    clip_eps = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.01
