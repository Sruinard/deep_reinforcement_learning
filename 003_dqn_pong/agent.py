import collections
import dataclasses
import functools
from dataclasses import dataclass
from typing import List, Callable
import time

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import random
import jax.random as jrandom
import optax
from flax.training import train_state, checkpoints
import net
import config
import flax
import loss_function


class TrainState(train_state.TrainState):
    target_params: flax.core.FrozenDict


class DQNAgent:

    def __init__(self, hparams: config.HParams, obs_shape: int = (1, 4, 84, 84, 1)):
        self.hparams = hparams
        self._brain = net.Brain(hparams=hparams)
        self.obs_shape = obs_shape
        self.policy = jax.jit(self.policy)
        self.update_step = jax.jit(self.update_step)

    def init_states(self) -> TrainState:
        params = self._brain.init(jax.random.PRNGKey(
            0), jnp.zeros(self.obs_shape))["params"]
        optimizer = optax.adam(learning_rate=self.hparams.learning_rate)
        return TrainState.create(apply_fn=self._brain.apply, params=params, tx=optimizer, target_params=params)

    def policy(self, state: TrainState, observation: jax.Array, n_actions: int, epsilon: float, rng: jrandom.PRNGKey):
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

    def collect_step(self, state: TrainState, env: gym.Env, observation: jax.Array, rng):
        epsilon = self.get_epsilon(state.step)
        action = self.policy(state, observation,
                             self.hparams.n_actions, epsilon, rng)
        next_observation, reward, is_done, _, _ = env.step(action)
        return observation, action, reward, is_done, next_observation

    def _loss_fn(self, params, target_params, batch, discount_t):
        obs_tm1, a_tm1, r_t, is_done, obs_t = batch
        q_tm1 = self._brain.apply({"params": params}, obs_tm1)
        q_t = self._brain.apply({"params": target_params}, obs_t)
        a_t = jnp.argmax(q_t, axis=1)
        return loss_function.double_q_learning_loss(
            q_tm1=q_tm1,
            q_t=q_t,
            a_tm1=a_tm1,
            a_t=a_t,
            r_t=r_t,
            discount_t=discount_t,
            is_done=is_done
        )

    def update_step(self, batch, state: TrainState, discount_t):
        loss, grads = jax.value_and_grad(self._loss_fn)(
            state.params,
            state.target_params,
            batch,
            discount_t=discount_t
        )

        state = state.apply_gradients(grads=grads)
        return (loss, grads), state

    def sync_target_network(self, state: TrainState):
        return state.replace(target_params=state.params)


def save_model(state: TrainState, path: str):
    checkpoints.save_checkpoint(
        path, target=state, step=state.step, keep=5, overwrite=True)


def load_model(path: str, state):
    return checkpoints.restore_checkpoint(path, target=state)
