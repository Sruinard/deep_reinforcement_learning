import functools

import flax
import gymnasium as gym
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
from flax.training import checkpoints, train_state
import rlax

import config
import net


class DQNAgent:

    def __init__(self, hparams: config.HParams, obs_shape: int = (1, 4)):
        self.hparams = hparams
        self._dqn = net.DQN(n_actions=hparams.n_actions)
        self.obs_shape = obs_shape
        self.policy = jax.jit(self.policy)
        self.update_step = jax.jit(self.update_step)

    def init_states(self) -> train_state.TrainState:
        params = self._dqn.init(jax.random.PRNGKey(
            0), jnp.zeros(self.obs_shape))["params"]
        optimizer = optax.adam(learning_rate=self.hparams.learning_rate)
        return train_state.TrainState.create(apply_fn=self._dqn.apply, params=params, tx=optimizer)

    def policy(self, state: train_state.TrainState, observation: jax.Array, n_actions: int, epsilon: float, rng: jrandom.PRNGKey):
        prob_of_selecting_random_action = jrandom.uniform(rng)
        f_explore = functools.partial(
            jax.random.randint, shape=(), minval=0, maxval=n_actions)
        q_values = state.apply_fn(
            {"params": state.params}, jnp.expand_dims(observation, axis=0))
        action = jax.lax.cond(prob_of_selecting_random_action <
                              epsilon, rng, f_explore, q_values, jnp.argmax)
        return action

    def get_epsilon(self, step_idx):
        delta = step_idx / self.hparams.epsilon_decay_steps
        return max(self.hparams.epsilon_end, self.hparams.epsilon_start - delta)

    def collect_step(self, state: train_state.TrainState, env: gym.Env, observation: jax.Array, rng):
        epsilon = self.get_epsilon(state.step)
        action = self.policy(state, observation,
                             self.hparams.n_actions, epsilon, rng)
        next_observation, reward, is_done, _, _ = env.step(int(action))
        return observation, action, reward, is_done, next_observation

    def _loss_fn(self, params, batch):
        obs_tm1, a_tm1, r_t, is_done, obs_t = batch
        q_tm1 = self._dqn.apply({"params": params}, obs_tm1)
        q_value_pred = q_tm1[jnp.arange(q_tm1.shape[0]), a_tm1]

        return jnp.mean(rlax.l2_loss(jax.lax.stop_gradient(r_t) - q_value_pred))

    def update_step(self, batch, state: train_state.TrainState):
        loss, grads = jax.value_and_grad(self._loss_fn)(
            state.params,
            batch
        )

        state = state.apply_gradients(grads=grads)
        return loss, state


def save_model(state: train_state.TrainState, path: str):
    checkpoints.save_checkpoint(
        path, target=state, step=state.step, keep=5, overwrite=True)


def load_model(path: str, state):
    return checkpoints.restore_checkpoint(path, target=state)
