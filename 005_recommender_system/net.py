import flax.linen as nn


class DQN(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            features=64, kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.relu(x)
        x = nn.Dense(
            features=64, kernel_init=nn.initializers.kaiming_normal())(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_actions)(x)
        return x
