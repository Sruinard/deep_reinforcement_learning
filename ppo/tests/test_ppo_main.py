import pytest

import jax
from .. import ppo_main
import envpool


def test_collect_trajectory_and_store_in_buffer():
    train_config = ppo_main.TrainConfig(
        num_envs=1,
        horizon=16
    )       
    env = envpool.make(
        "CartPole-v1",
        env_type="gym",
        num_envs=1,
    )
    buffer = ppo_main.Buffer.create(
        horizon=train_config.horizon,
        observation_shape=(4,)
    )
    state = ppo_main.create_train_state(
        train_config, env.observation_space.shape, env.action_space.n)
    rng = jax.random.PRNGKey(train_config.model_seed)
    collected_buffer = ppo_main.collect_trajectory(state, buffer, env, rng, train_config)
    assert collected_buffer.obs.shape == (train_config.horizon + 1, 4)
    # assert all arrays in obs have non-zero values
    assert all(collected_buffer.obs.any(axis=1))
    