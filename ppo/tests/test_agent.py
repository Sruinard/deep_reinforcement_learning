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


@pytest.mark.trajectories
def test_collect_trajectory(envs: gym.Env, num_envs: int):
    """Test the collect_trajectory function."""
    horizon = 32

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


@pytest.mark.trajectories
def test_advantage_from_rollout():
    num_envs = 10
    horizon = 32
    trajectory = agent.Trajectory(
        obs=jnp.zeros((horizon, num_envs,  4)),
        actions=jnp.zeros((horizon, num_envs)),
        rewards=jnp.zeros((horizon, num_envs)),
        dones=jnp.zeros((horizon, num_envs)),
        log_probs=jnp.zeros((horizon, num_envs)),
        values=jnp.zeros((horizon, num_envs)),
        advantages=jnp.zeros((horizon, num_envs)),
        returns=jnp.zeros((horizon, num_envs))
    )

    discount_factor = 0.99
    lambda_ = 0.2

    advantages, returns = agent.compute_gae_from_trajectory(
        trajectory.rewards, trajectory.values, trajectory.dones, discount_factor, lambda_)

    assert advantages.shape == (horizon - 1, num_envs)
    assert returns.shape == (horizon - 1, num_envs)


@pytest.mark.update
def test_agent_can_collect_and_update():
    """Test that the agent can collect and update."""
    num_envs = 10
    horizon = 32
    envs = envpool.make("CartPole-v1", env_type="gym", num_envs=num_envs)

    ppo_net = ppo_model.PPOActorCritic(envs.action_space.n)

    state = train_state.TrainState.create(
        apply_fn=ppo_net.apply,
        params=ppo_net.init(
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

    gamma = 0.99
    gae_lambda = 0.2
    rng = jax.random.PRNGKey(0)

    state, loss, rng = agent.collect_and_update_ppo_loop(
        envs=envs, state=state, trajectory=trajectory, horizon=horizon, gamma=gamma, gae_lambda=gae_lambda, rng=rng)
    assert loss is not None
