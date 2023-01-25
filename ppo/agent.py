
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


@partial(jax.vmap, in_axes=(None, 0, None, None, None))
def ppo_forward(state: train_state.TrainState, obs, rng: jax.random.PRNGKey, n_actions: int, is_deterministic: bool = False):
    """Collect a single trajectory."""
    rng, key = jax.random.split(rng)
    logits, value = state.apply_fn({"params": state.params}, obs)
    probs = jax.nn.softmax(logits)
    action = jax.lax.select(is_deterministic, jnp.argmax(logits), jax.random.choice(
        key, a=jnp.arange(n_actions), p=probs, replace=False))

    log_probs = jax.nn.log_softmax(logits)[action]
    return action, value.squeeze(), log_probs


def collect_trajectory(envs, state, trajectory, horizon, rng, is_deterministic=False):
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
    return state, trajectory, rng

