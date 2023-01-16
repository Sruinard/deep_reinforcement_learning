import dataclasses


@dataclasses.dataclass
class HParams:
    n_actions: int

    log_dir: str = "./logs/"
    checkpoints_dir: str = "./checkpoints/"
    n_train_episodes: int = 2000
    evaluate_every_n_epoch: int = 1
    n_episodes_per_eval: int = 1
    batch_size: int = 32
    learning_rate: float = 0.0001
    memory_size: int = 4000
    gamma: float = 0.99
    epsilon_decay_steps: int = 10000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.03
    seed: int = 42
    sync_every_n_steps: int = 50

    hidden_layer_n_neurons: int = 512
