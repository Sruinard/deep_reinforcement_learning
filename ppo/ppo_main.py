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
import dataclasses


@dataclasses.dataclass
class TrainConfig:
    num_envs: int = 1
    env_seed: int = 42

    n_episodes: int = 100000
    n_updates_per_rollout: int = 4
    horizon: int = 500
    mini_batch_size: int = 4

    learning_rate: float = 0.003
    model_seed: int = 42

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coeff: float = 0.01
    critic_loss_coeff: float = 0.5
    log_every: int = 10


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

    @classmethod
    def create(cls, horizon, num_envs, observation_shape):
        return cls(
            obs=jnp.zeros((horizon + 1, num_envs, *observation_shape)),
            actions=jnp.zeros((horizon + 1, num_envs)),
            rewards=jnp.zeros((horizon + 1, num_envs)),
            dones=jnp.zeros((horizon + 1, num_envs)),
            log_probs=jnp.zeros((horizon + 1, num_envs)),
            values=jnp.zeros((horizon + 1, num_envs)),
            advantages=jnp.zeros((horizon, num_envs)),
            target_returns=jnp.zeros((horizon, num_envs)),
            episode_returns=jnp.zeros((1, num_envs))
        )

    # double method for readability.
    @classmethod
    def reset(cls, horizon, num_envs, observation_shape):
        return cls(
            obs=jnp.zeros((horizon + 1, num_envs, *observation_shape)),
            actions=jnp.zeros((horizon + 1, num_envs)),
            rewards=jnp.zeros((horizon + 1, num_envs)),
            dones=jnp.zeros((horizon + 1, num_envs)),
            log_probs=jnp.zeros((horizon + 1, num_envs)),
            values=jnp.zeros((horizon + 1, num_envs)),
            advantages=jnp.zeros((horizon, num_envs)),
            target_returns=jnp.zeros((horizon, num_envs)),
            episode_returns=jnp.zeros((1, num_envs))
        )


def store_step_in_buffer(buffer: Buffer, step: int, obs, action, reward, done, log_prob, value):
    buffer = buffer.replace(
        obs=buffer.obs.at[step].set(obs),
        actions=buffer.actions.at[step].set(action),
        rewards=buffer.rewards.at[step].set(reward),
        dones=buffer.dones.at[step].set(done),
        log_probs=buffer.log_probs.at[step].set(log_prob),
        # convert num_envs x value_estimate --> value_estimate x num_envs
        values=buffer.values.at[step].set(jnp.squeeze(value.T))
    )
    return buffer


def compute_generalized_advantage_estimates(buffer: Buffer, gamma: float, gae_lambda: float) -> Buffer:
    """Compute generalized advantage estimates."""
    discount_t = gamma * (1 - buffer.dones[:-1])

    gae = rlax.truncated_generalized_advantage_estimation(
        r_t=buffer.rewards[:-1],
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

    buffer = buffer.replace(actions=jnp.array(buffer.actions, dtype=jnp.int32))

    # https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax_scan.py
    def flatten(x):
        return jnp.array(x.reshape((-1,) + x.shape[1:]), dtype=x.dtype)
    batch = jax.tree_map(flatten, buffer)
    return batch

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


def create_train_state(train_config: TrainConfig, observation_shape, num_actions):
    """Create train state."""
    actor_critic = ActorCritic(num_actions=num_actions)
    rng = jax.random.PRNGKey(train_config.model_seed)
    obs = jnp.zeros((1, *observation_shape))
    initial_params = actor_critic.init(rng, obs)["params"]
    optimizer = optax.adam(train_config.learning_rate)
    return train_state.TrainState.create(
        apply_fn=actor_critic.apply,
        params=initial_params,
        tx=optimizer
    )


@jax.jit
def ppo_net(train_state: train_state.TrainState, params, obs):
    """Get action logits and value estimates."""
    logits, value_estimate = train_state.apply_fn({"params": params}, obs)
    return logits, value_estimate


@jax.jit
def policy_from_logits(logits, rng, is_training=True):
    exploit = jnp.argmax(logits, axis=-1)
    explore = jax.random.categorical(rng, logits, axis=-1)
    action = jax.lax.select(is_training, explore, exploit)
    return action


def collect_trajectory(train_state: train_state.TrainState, buffer: Buffer, env: gym.Env, rng: jax.random.PRNGKey, train_config: TrainConfig):
    """Collect a single trajectory."""
    obs, _ = env.reset()
    for step in range(train_config.horizon + 1):  # +1 for the last step
        rng, rng_step = jax.random.split(rng)
        logits, value_estimate = ppo_net(train_state, train_state.params, obs)
        action = policy_from_logits(logits, rng_step, is_training=True)
        next_obs, reward, done, _, _ = env.step(np.asarray(action))
        log_prob = on_policy_log_prob(logits, action)
        buffer = store_step_in_buffer(
            buffer=buffer,
            step=step,
            obs=obs,
            action=action,
            reward=reward,
            done=done,
            log_prob=log_prob,
            value=value_estimate
        )
        obs = next_obs
        # envpool has autoreset as default, so we don't need to handle "if done"
        # if done:
        #     obs, _ = env.reset()
    return buffer


@partial(jax.vmap, in_axes=(0, 0))
def on_policy_log_prob(logits, action):
    log_probs = nn.log_softmax(logits)[action]
    return log_probs


@jax.jit
def loss_fn(state: train_state.TrainState, params, buffer: Buffer, clip_eps: float, critic_loss_coeff: float, entropy_coeff: float):
    """Compute loss."""
    logits, value_estimates = state.apply_fn(
        {"params": params}, buffer.obs[:-1])

    # Actor loss
    log_probs = on_policy_log_prob(logits, buffer.actions[:-1])
    prob_ratio = jnp.exp(log_probs - buffer.log_probs[:-1])
    normalized_advantage = (buffer.advantages - buffer.advantages.mean()) / (
        buffer.advantages.std() + 1e-8)
    actor_loss = rlax.clipped_surrogate_pg_loss(
        prob_ratios_t=prob_ratio,
        adv_t=normalized_advantage,
        epsilon=clip_eps
    )

    # critic loss
    critic_loss = jnp.mean(rlax.huber_loss(
        buffer.target_returns - value_estimates
    ))

    # entropy loss
    entropy_loss = rlax.entropy_loss(
        logits, entropy_coeff * jnp.ones_like(buffer.actions[:-1]))

    loss = actor_loss + critic_loss_coeff * critic_loss - entropy_loss
    return loss, (actor_loss, critic_loss, entropy_loss)


def update_ppo_model(state: train_state.TrainState, buffer: Buffer, train_config: TrainConfig):
    """Update PPO model."""

    batch = get_batch(buffer, train_config)
    (loss, (actor_loss, critic_loss, entropy_loss)), grads = jax.value_and_grad(
        loss_fn, argnums=1, has_aux=True)(state, state.params, batch, train_config.clip_eps, train_config.critic_loss_coeff, train_config.entropy_coeff)
    state = state.apply_gradients(grads=grads)

    return state, (loss, actor_loss, critic_loss, entropy_loss)


# create run_loop
def run_loop(train_config: TrainConfig, env: gym.Env, state: train_state.TrainState):
    """Run training loop."""
    rng = jax.random.PRNGKey(train_config.model_seed)
    buffer = Buffer.create(horizon=train_config.horizon,
                           observation_shape=env.observation_space.shape)
    average_reward = 0
    for episode in range(1, train_config.n_episodes + 1):
        rng, rng_collect = jax.random.split(rng)
        buffer = collect_trajectory(
            state, buffer, env, rng_collect, train_config)
        for _ in range(train_config.n_updates_per_rollout):
            state, (loss, actor_loss, critic_loss, entropy_loss) = update_ppo_model(
                state, buffer, train_config)

        average_reward_per_trajectory = buffer.rewards.sum() / max(buffer.dones.sum(), 1)
        average_reward += average_reward_per_trajectory

        if episode % train_config.log_every == 0:
            print(
                f"Episode: {episode}, loss: {loss}, actor_loss: {actor_loss}, critic_loss: {critic_loss}, entropy_loss: {entropy_loss}")
            print(f"Average reward: {average_reward / train_config.log_every}")
            average_reward = 0
        buffer = buffer.reset(train_config.horizon,
                              env.observation_space.shape)


if __name__ == "__main__":
    train_config = TrainConfig()
    env = envpool.make(
        "CartPole-v1",
        env_type="gym",
        num_envs=train_config.num_envs,
    )
    state = create_train_state(
        train_config, env.observation_space.shape, env.action_space.n)
    run_loop(train_config, env, state)
