
from typing import List
import rlax
import gymnasium as gym
import envpool
from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import flax
from functools import partial


@flax.struct.dataclass
class Trajectory:
    """A single trajectory."""
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray

    log_probs: jnp.ndarray
    values: jnp.ndarray
    advantages: jnp.ndarray = None
    returns: jnp.ndarray = None


@partial(jax.vmap, in_axes=(None, 0, None, None, None))
def ppo_forward(state: train_state.TrainState, obs, rng: jax.random.PRNGKey, n_actions: int, is_deterministic: bool = False) -> List[jnp.ndarray]:
    """Collect a single trajectory."""
    rng, key = jax.random.split(rng)
    logits, value = state.apply_fn({"params": state.params}, obs)
    probs = jax.nn.softmax(logits)
    action = jax.lax.select(is_deterministic, jnp.argmax(logits), jax.random.choice(
        key, a=jnp.arange(n_actions), p=probs, replace=False))

    log_probs = jax.nn.log_softmax(logits)[action]
    return action, value.squeeze(), log_probs


@partial(jax.vmap, in_axes=(None, 0, 0))
def ppo_forward_during_update_steps(state: train_state.TrainState, obs, action) -> List[jnp.ndarray]:
    """Collect a single trajectory."""
    logits, value = state.apply_fn({"params": state.params}, obs)
    log_probs = jax.nn.log_softmax(logits)[action]
    return logits, value.squeeze(), log_probs


def collect_trajectory(envs: gym.Env, state, trajectory: Trajectory, horizon, rng, is_deterministic=False):
    """Collect a single trajectory."""
    rng, ppo_key = jax.random.split(rng)
    obs, _ = envs.reset()
    # add 1 to horizon to include the last step needed for computing GAE.
    for step in range(horizon + 1):
        action, value, log_probs = ppo_forward(
            state, obs, ppo_key, envs.action_space.n, is_deterministic)
        next_obs, reward, done, _, _ = envs.step(np.array(action))
        trajectory = trajectory.replace(
            obs=trajectory.obs.at[step].set(obs),
            actions=trajectory.actions.at[step].set(action),
            rewards=trajectory.rewards.at[step].set(reward),
            dones=trajectory.dones.at[step].set(done),
            log_probs=trajectory.log_probs.at[step].set(log_probs),
            values=trajectory.values.at[step].set(value)
        )
        obs = next_obs
    return state, trajectory, rng


@partial(jax.vmap, in_axes=(1, 1, 1, None, None), out_axes=1)
def compute_gae_from_trajectory(rewards, values, dones, gamma: float, gae_lambda: float):
    """Compute GAE."""

    gamma = gamma * (1 - dones[1:])
    rewards = rewards[1:]

    advantages = rlax.truncated_generalized_advantage_estimation(
        r_t=rewards,
        discount_t=gamma,
        lambda_=gae_lambda,
        values=values
    )
    returns = advantages + values[:-1]
    return advantages, returns


def collect_experiences(envs: gym.Env, state, trajectory: Trajectory, horizon, rng, gamma: float, gae_lambda: float, is_deterministic=False):
    """Collect a single trajectory."""
    state, trajectory, rng = collect_trajectory(
        envs, state, trajectory, horizon, rng, is_deterministic)
    advantages, returns = compute_gae_from_trajectory(
        trajectory.rewards, trajectory.values, trajectory.dones, gamma, gae_lambda)
    trajectory = trajectory.replace(advantages=advantages, returns=returns)
    return trajectory, rng


def ppo_loss(state: train_state.TrainState, trajectory: Trajectory, entropy_coeff: float = 0.01, clip_range: float = 0.2):
    """Compute the PPO loss."""

    logits, values, new_log_probs = ppo_forward_during_update_steps(
        state, trajectory.obs, jnp.asarray(trajectory.actions, dtype=jnp.int32))

    entropy = rlax.entropy_loss(logits, entropy_coeff *
                                jnp.ones_like(trajectory.actions))
    old_log_probs = trajectory.log_probs
    prob_ratios_t = jnp.exp(new_log_probs - old_log_probs)

    normalized_advantages = (trajectory.advantages - jnp.mean(
        trajectory.advantages)) / (jnp.std(trajectory.advantages) + 1e-8)
    actor_loss = rlax.clipped_surrogate_pg_loss(
        prob_ratios_t=prob_ratios_t,
        adv_t=normalized_advantages,
        epsilon=clip_range
    )
    critic_loss = jnp.mean(rlax.l2_loss(trajectory.returns - values.squeeze()))
    loss = actor_loss + 0.5 * critic_loss - entropy
    return loss


def ppo_update(state: train_state.TrainState, trajectory: Trajectory):
    """Update the PPO agent."""
    loss, grads = jax.value_and_grad(
        ppo_loss, allow_int=True)(state, trajectory)
    state = state.apply_gradients(grads=grads)
    return state, loss


def collect_and_update_ppo_loop(envs: gym.Env, state, trajectory: Trajectory, horizon, rng, gamma: float, gae_lambda: float, **kwargs):
    """Collect a single trajectory."""
    trajectory, rng = collect_experiences(
        envs, state, trajectory, horizon, rng, gamma, gae_lambda)

    # r + gamma * value(next_state, next_action) - value(state, action) <-- next_state is why we need an additional element to complete
    # the computation of the GAE.

    trajectory = trajectory.replace(
        # Remove last element as GAE computation requires one more element and otherwise the shapes are not compatible in the loss computation.
        obs=trajectory.obs[:-1].reshape(-1, trajectory.obs.shape[-1]),
        actions=trajectory.actions[:-1].reshape(-1),
        rewards=trajectory.rewards[:-1].reshape(-1),
        dones=trajectory.dones[:-1].reshape(-1),
        log_probs=trajectory.log_probs[:-1].reshape(-1),
        values=trajectory.values[:-1].reshape(-1),
        # advantages and returns have already one element less because of the GAE computation.
        advantages=trajectory.advantages.reshape(-1),
        returns=trajectory.returns.reshape(-1)
    )
    state, loss = ppo_update(state, trajectory)
    return state, loss, rng
