import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import rlax
import flax
from flax.training import checkpoints, train_state
import gymnasium as gym
import envpool
from functools import partial
import numpy as np
from typing import List


class TrainConfig:
    num_envs = 32
    env_seed = 42

    n_epochs = 100000
    n_updates_per_rollout = 4
    horizon = 64
    mini_batch_size = 4

    learning_rate = 0.003
    model_seed = 42

    gamma = 0.99
    gae_lambda = 0.95
    clip_eps = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.01


@flax.struct.dataclass
class Buffer:
    """A single trajectory."""
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray

    log_probs: jnp.ndarray
    values: jnp.ndarray
    advantages: jnp.ndarray = None
    target_returns: jnp.ndarray = None
    episode_returns: jnp.ndarray = None

    def __init__(self, horizon, observation_shape) -> None:
        self.horizon = horizon
        self.observation_shape = observation_shape
        self.reset(horizon, observation_shape)

    def reset(self, horizon, observation_shape):
        self.obs = jnp.zeros((horizon + 1, *observation_shape))
        self.actions = jnp.zeros((horizon + 1))
        self.rewards = jnp.zeros((horizon + 1))
        self.dones = jnp.zeros((horizon + 1))

        self.log_probs = jnp.zeros((horizon + 1))
        self.values = jnp.zeros((horizon + 1))

        # advantags and target returns require a one step lookahead
        self.advantages = jnp.zeros((horizon))
        self.target_returns = jnp.zeros((horizon))
        self.episode_returns = jnp.zeros((1,))


def store_step_in_buffer(buffer: Buffer, step: int, obs, action, reward, done, log_prob, value):
    buffer.replace(
        obs=buffer.obs.at[step].set(obs),
        actions=buffer.actions.at[step].set(action),
        rewards=buffer.rewards.at[step].set(reward),
        dones=buffer.dones.at[step].set(done),
        log_probs=buffer.log_probs.at[step].set(log_prob),
        values=buffer.values.at[step].set(value)
    )
    return buffer


def compute_generalized_advantage_estimates(buffer: Buffer, gamma: float, gae_lambda: float) -> Buffer:
    """Compute generalized advantage estimates."""
    discount_t = gamma * (1 - buffer.dones[:-1])

    gae = rlax.truncated_generalized_advantage_estimation(
        rewards=buffer.rewards[:-1],
        discount_t=discount_t,
        lambda_=gae_lambda,
        values=buffer.values,
    )
    buffer = buffer.replace(advantages=gae)
    return buffer


def compute_target_returns(buffer: Buffer, gamma: float, n_step_bootstrap=3) -> Buffer:
    """Compute target returns."""
    discount_t = gamma * (1 - buffer.dones[:-1])
    target_returns = rlax.n_step_bootstrapped_returns(
        r_t=buffer.rewards[1:],
        discount_t=discount_t,
        v_t=buffer.values[1:],
        n=n_step_bootstrap,
        lambda_t=1.0,
        stop_target_gradients=True
    )
    buffer = buffer.replace(target_returns=target_returns)
    return buffer


def get_batch(buffer: Buffer, train_config: TrainConfig):
    """Get batch from buffer."""
    buffer = compute_generalized_advantage_estimates(
        buffer=buffer,
        gamma=train_config.gamma,
        gae_lambda=train_config.gae_lambda
    )
    buffer = compute_target_returns(
        buffer=buffer,
        gamma=train_config.gamma
    )
    return (
        buffer.obs[:-1],
        buffer.actions[:-1],
        buffer.advantages,
        buffer.target_returns,
        buffer.log_probs[:-1]
    )

##########################
######## Agent ###########
##########################

class ActorCritic(nn.Module):
    num_actions: int

    @nn.compact
    def __call__(self, obs):
        x = nn.Sequential([
            nn.Dense(64),
            nn.leaky_relu,
            nn.Dense(64),
            nn.leaky_relu
        ])(obs)

        logits = nn.Dense(self.num_actions)(x)
        value_estimate = nn.Dense(1)(x)
        return logits, value_estimate

def policy_from_logits(logits, rng, is_training=True):
    exploit = jnp.argmax(logits, axis=-1)
    explore = jax.random.categorical(rng, logits, axis=-1)
    action = jax.lax.select(is_training, explore, exploit)
    return action

