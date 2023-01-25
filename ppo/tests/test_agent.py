import agent
import ppo_model
import jax.numpy as jnp
import envpool
import gymnasium as gym
from flax.training import train_state
import jax
import pytest
import optax


@pytest.fixture
def num_envs() -> int:
    """Fixture for the number of environments."""
    return 10


@pytest.fixture
def envs(num_envs) -> gym.Env:
    """Fixture for the environment."""
    return envpool.make("CartPole-v1", env_type="gym", num_envs=num_envs)


def test_collect_trajectory(envs: gym.Env, num_envs: int):
    """Test the collect_trajectory function."""
    horizon = 10

    state = train_state.TrainState.create(
        apply_fn=ppo_model.PPOActorCritic(envs.action_space.n).apply,
        params=ppo_model.PPOActorCritic(envs.action_space.n).init(
            jax.random.PRNGKey(0), jnp.zeros((1, envs.observation_space.shape[0])))["params"],
        tx=optax.adam(1e-3),
    )

    trajectory = agent.Trajectory(
        obs=jnp.zeros((horizon, num_envs,  envs.observation_space.shape[0])),
        actions=jnp.zeros((horizon, num_envs)),
        rewards=jnp.zeros((horizon, num_envs)),
        dones=jnp.zeros((horizon, num_envs)),
        log_probs=jnp.zeros((horizon, num_envs)),
        values=jnp.zeros((horizon, num_envs)),
        advantages=jnp.zeros((horizon, num_envs)),
        returns=jnp.zeros((horizon, num_envs))
    )

    state, trajectory, rng = agent.collect_trajectory(
        envs=envs, state=state, trajectory=trajectory, horizon=horizon, rng=jax.random.PRNGKey(0))
    assert trajectory.obs.shape == (
        horizon, num_envs, envs.observation_space.shape[0])

    # assert non-zeros
    assert jnp.any(trajectory.obs)

    assert trajectory.actions.shape == (horizon, num_envs)
    assert trajectory.rewards.shape == (horizon, num_envs)
    assert trajectory.dones.shape == (horizon, num_envs)
    assert trajectory.log_probs.shape == (horizon, num_envs)
    assert trajectory.values.shape == (horizon, num_envs)
