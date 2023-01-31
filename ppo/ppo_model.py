import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import rlax
from flax.training import checkpoints, train_state


class PPOActorCritic(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, x):
        x = nn.Sequential([
            nn.Dense(64),
            nn.relu,
            nn.Dense(64),
            nn.relu,
        ])(x)

        logits = nn.Dense(self.num_actions)(x)
        value = nn.Dense(1)(x)
        return logits, value
