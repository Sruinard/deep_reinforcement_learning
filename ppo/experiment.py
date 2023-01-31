
import gymnasium as gym
import jax
import jax.numpy as jnp
import envpool
import rlax
import optax
from flax.training import train_state

import ppo_model
import environment
import agent
import config


def run():
    # setup
    rng = jax.random.PRNGKey(config.Config.model_seed)
    envs = envpool.make("CartPole-v1", env_type="gym",
                        num_envs=config.Config.n_envs, seed=config.Config.env_seed
                        )

    # model
    model = ppo_model.PPOActorCritic(envs.action_space.n)
    # optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip gradients at norm 1
        optax.adam(learning_rate=config.Config.learning_rate)
    )

    # initialize state
    rng, model_rng = jax.random.split(rng)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(model_rng, jnp.ones(
            (1, envs.observation_space.shape[0])))["params"],
        tx=optimizer,
    )

    # init trajectory
    trajectory = agent.Trajectory(
        obs=jnp.zeros((config.Config.horizon, config.Config.n_envs,
                      envs.observation_space.shape[0])),
        actions=jnp.zeros((config.Config.horizon, config.Config.n_envs)),
        rewards=jnp.zeros((config.Config.horizon, config.Config.n_envs)),
        dones=jnp.zeros((config.Config.horizon, config.Config.n_envs)),
        log_probs=jnp.zeros((config.Config.horizon, config.Config.n_envs)),
        values=jnp.zeros((config.Config.horizon, config.Config.n_envs)),
        advantages=jnp.zeros((config.Config.horizon, config.Config.n_envs)),
        returns=jnp.zeros((config.Config.horizon, config.Config.n_envs)),
        episode_returns=jnp.zeros((config.Config.n_envs))
    )

    for episode in range(config.Config.n_epochs):
        rng, update_rng = jax.random.split(rng)
        state, (loss, actor_loss, critic_loss, entropy), trajectory, rng = agent.collect_and_update_ppo_loop(
            envs=envs,
            state=state,
            trajectory=trajectory,
            rng=update_rng,
            n_updates_per_rollout=config.Config.n_updates_per_rollout,
            horizon=config.Config.horizon,
            n_envs=config.Config.n_envs,
            gamma=config.Config.gamma,
            gae_lambda=config.Config.gae_lambda,
            clip_eps=config.Config.clip_eps,
            value_loss_coef=config.Config.value_loss_coef,
            entropy_coef=config.Config.entropy_coef,
            batch_size=config.Config.n_envs,
            mini_batch_size=config.Config.mini_batch_size,
            verbose=False
        )
        print(f"""
            Episode: {episode} ---
            Mean Action Probabitilities from log_probs: {jnp.mean(jnp.exp(trajectory.log_probs))} ---
            Loss: {loss} ---
            Actor Loss: {actor_loss} ---
            Critic Loss: {critic_loss} ---
            Entropy: {entropy} ---
            actions: n_ones: {jnp.sum(trajectory.actions)} ---
            actions: n_zeros: {jnp.sum(1 - trajectory.actions)} ---
            Mean Episode Return: {trajectory.episode_returns.mean()} ---
            Episode Return: {trajectory.episode_returns} ---
        """)
        trajectory = agent.Trajectory(
            obs=jnp.zeros((config.Config.horizon, config.Config.n_envs,
                           envs.observation_space.shape[0])),
            actions=jnp.zeros((config.Config.horizon, config.Config.n_envs)),
            rewards=jnp.zeros((config.Config.horizon, config.Config.n_envs)),
            dones=jnp.zeros((config.Config.horizon, config.Config.n_envs)),
            log_probs=jnp.zeros((config.Config.horizon, config.Config.n_envs)),
            values=jnp.zeros((config.Config.horizon, config.Config.n_envs)),
            advantages=jnp.zeros(
                (config.Config.horizon, config.Config.n_envs)),
            returns=jnp.zeros((config.Config.horizon, config.Config.n_envs)),
            episode_returns=jnp.zeros((config.Config.n_envs))
        )


if __name__ == "__main__":
    run()
