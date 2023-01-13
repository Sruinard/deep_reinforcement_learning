class TrainConfig:
    def __init__(
        self,
        obs_shape: tuple,
        n_actions: int,
        n_train_episodes: int = 1,
        n_eval_episodes: int = 1,
        eval_every_n_episodes: int = 20,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        beta: float = 0.01,
        entropy_coeff: float = 0.01,
        max_trajectory_length: int = 500,
        n_steps: int = 20,
        seed: int = 42
    ):
        self.obs_shape = obs_shape
        self.n_actions = n_actions

        self.n_train_episodes = n_train_episodes
        self.n_eval_episodes = n_eval_episodes
        self.eval_every_n_episodes = eval_every_n_episodes
        self.learning_rate = learning_rate

        # for creating targets and weighting loss
        self.max_trajectory_length = max_trajectory_length
        self.beta = beta
        self.entropy_coeff = entropy_coeff
        self.n_steps = n_steps
        self.gamma = gamma

        # for reproducibility
        self.seed = seed
