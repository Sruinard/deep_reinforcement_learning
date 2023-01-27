import gymnasium as gym
import jax
import jax.numpy as jnp
import envpool
import rlax
import optax
from flax.

import ppo_model
import environment
import agent
import config

def test_model_can_train_on_cartpole():
    # setup
    env_name = "CartPole-v1"
    env_seed = 0
    env = environment.JaxEnv(env_name, env_seed)
    envs = gym.vector.make(env_name, num_envs=1)

    # model
    model = ppo_model.PPOActorCritic(env.action_space.n)

    # optimizer
    optimizer = optax.adam(config.Config.learning_rate)

    # agent
    agent.collect_and_update_ppo_loop(

    )

    # train
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(jax.random.PRNGKey(0), jnp.zeros((1, num_observations)))["params"],
        tx=optimizer,
    )

    trajectory = agent.Trajectory(
        obs=jnp.zeros((horizon, num_envs, num_observations)),
        actions=jnp.zeros((horizon, num_envs)),
        rewards=jnp.zeros((horizon, num_envs)),
        dones=jnp.zeros((horizon, num_envs)),
        log_probs=jnp.zeros((horizon, num_envs)),
        values=jnp.zeros((horizon, num_envs)),
        advantages=jnp.zeros((horizon, num_envs)),
        returns=jnp.zeros((horizon, num_envs))
    )

    for _ in range(num_timesteps):
        state, trajectory, rng = agent_.collect_trajectory(
            envs=envs, state=state, trajectory=trajectory, horizon=horizon, rng=jax.random.PRNGKey(0
