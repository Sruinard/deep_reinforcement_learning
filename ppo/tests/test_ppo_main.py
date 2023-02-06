import pytest

import jax
import jax.numpy as jnp
from .. import ppo_main
import envpool


# parameterize the number of environments
@pytest.mark.parametrize("num_envs", [1, 2, 4])
def test_collect_trajectory_and_store_in_buffer(num_envs):
    train_config = ppo_main.TrainConfig(
        num_envs=num_envs,
        horizon=16
    )       
    env = envpool.make(
        "CartPole-v1",
        env_type="gym",
        num_envs=train_config.num_envs,
    )
    buffer = ppo_main.Buffer.create(
        horizon=train_config.horizon,
        num_envs=train_config.num_envs,
        observation_shape=(4,)
    )
    state = ppo_main.create_train_state(
        train_config, env.observation_space.shape, env.action_space.n)
    rng = jax.random.PRNGKey(train_config.model_seed)
    collected_buffer = ppo_main.collect_trajectory(state, buffer, env, rng, train_config)
    assert collected_buffer.obs.shape == (train_config.horizon + 1, train_config.num_envs, 4)
    # assert observations are non-zero and hence replaced
    assert not jnp.allclose(collected_buffer.obs, 0.0)


    