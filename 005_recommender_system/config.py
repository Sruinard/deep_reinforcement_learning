import dataclasses


@dataclasses.dataclass
class HParams:
    n_actions: int

    log_dir: str = "./logs/"
    checkpoints_dir: str = "./checkpoints/"
    n_train_episodes: int = 10000
    evaluate_every_n_epoch: int = 100
    n_episodes_per_eval: int = 5
    batch_size: int = 64
    learning_rate: float = 0.0005
    memory_size: int = 1000
    gamma: float = 0.99
    epsilon_decay_steps: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.03
    seed: int = 42
    sync_every_n_steps: int = 50

    hidden_layer_n_neurons: int = 512
