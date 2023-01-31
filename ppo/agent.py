
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
    episode_returns: jnp.ndarray = None


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


def collect_trajectory(envs: gym.Env, state, trajectory: Trajectory, horizon, rng, is_deterministic=False):
    """Collect a single trajectory."""
    obs, _ = envs.reset()
    # add 1 to horizon to include the last step needed for computing GAE.
    for step in range(horizon + 1):
        rng, ppo_key = jax.random.split(rng)
        action, value, log_probs = ppo_forward(
            state, obs, ppo_key, envs.action_space.n, is_deterministic)
        next_obs, reward, done, _, _ = envs.step(
            np.array(action), env_id=np.arange(32))
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
    rewards = rewards[:-1]

    advantages = rlax.truncated_generalized_advantage_estimation(
        r_t=rewards,
        discount_t=gamma,
        lambda_=gae_lambda,
        values=values
    )
    returns = advantages + values[:-1]
    return advantages, returns


def compute_envpool_reward_per_env(rewards, dones):
    """Compute the average episode return per environment."""
    rewards = rewards[1:]
    dones = dones[1:]
    episode_returns = rewards.sum(
        axis=0) / (dones.sum(axis=0) + 1)  # always one episode
    return episode_returns


def collect_experiences(envs: gym.Env, state, trajectory: Trajectory, horizon, rng, gamma: float, gae_lambda: float, is_deterministic=False):
    """Collect a single trajectory."""
    rng, collect_trajectory_rng = jax.random.split(rng)
    state, trajectory, rng = collect_trajectory(
        envs, state, trajectory, horizon, collect_trajectory_rng, is_deterministic)
    advantages, returns = compute_gae_from_trajectory(
        trajectory.rewards, trajectory.values, trajectory.dones, gamma, gae_lambda)

    average_episode_return_per_env = compute_envpool_reward_per_env(
        trajectory.rewards, trajectory.dones)
    trajectory = trajectory.replace(
        advantages=advantages, returns=returns, episode_returns=average_episode_return_per_env)

    return trajectory, rng


@jax.jit
def ppo_update(state: train_state.TrainState, trajectory: Trajectory):
    """Update the PPO agent."""

    @partial(jax.vmap, in_axes=(None, 0, 0))
    def ppo_forward_during_update_steps(params, obs, action) -> List[jnp.ndarray]:
        """Collect a single trajectory."""
        logits, value = state.apply_fn({"params": params}, obs)
        log_probs = jax.nn.log_softmax(logits)[action]
        return logits, value.squeeze(), log_probs

    @jax.jit
    def ppo_loss(params, trajectory: Trajectory, entropy_coeff: float = 0.01, clip_range: float = 0.2):
        """Compute the PPO loss."""

        logits, values, new_log_probs = ppo_forward_during_update_steps(
            params, trajectory.obs, jnp.asarray(trajectory.actions, dtype=jnp.int32))

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
        critic_loss = jnp.mean(rlax.huber_loss(
            jax.lax.stop_gradient(trajectory.returns) - values.squeeze()))
        loss = actor_loss + 0.5 * critic_loss - entropy
        return loss, (actor_loss, critic_loss, entropy)

    ppo_forward_during_update_steps = jax.jit(ppo_forward_during_update_steps)
    ppo_loss = jax.jit(ppo_loss)

    (loss, (actor_loss, critic_loss, entropy)), grads = jax.value_and_grad(
        ppo_loss, has_aux=True, allow_int=True)(state.params, trajectory)
    state = state.apply_gradients(grads=grads)
    return state, (loss, actor_loss, critic_loss, entropy)


def collect_and_update_ppo_loop(envs: gym.Env, state, trajectory: Trajectory, horizon, rng, gamma: float, gae_lambda: float, batch_size: int, mini_batch_size: int, **kwargs):
    """Collect a single trajectory."""
    rng, experience_rng = jax.random.split(rng)
    trajectory, rng = collect_experiences(
        envs, state, trajectory, horizon, experience_rng, gamma, gae_lambda)

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
    for _ in range(kwargs.get("n_updates_per_rollout", 10)):

        # sample a
        rng, subkey = jax.random.split(rng)
        b_inds = jax.random.permutation(subkey, batch_size, independent=True)
        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            mb_inds = b_inds[start:end]
            b_trajectory = Trajectory(
                obs=trajectory.obs[mb_inds],
                actions=trajectory.actions[mb_inds],
                rewards=trajectory.rewards[mb_inds],
                dones=trajectory.dones[mb_inds],
                log_probs=trajectory.log_probs[mb_inds],
                values=trajectory.values[mb_inds],
                advantages=trajectory.advantages[mb_inds],
                returns=trajectory.returns[mb_inds],
                episode_returns=trajectory.episode_returns[mb_inds]
            )

            state, (loss, actor_loss, critic_loss, entropy) = ppo_update(
                state, b_trajectory)
            if kwargs.get("verbose", False):
                print(f"Loss: {loss}")

    return state, (loss, actor_loss, critic_loss, entropy), trajectory, rng
