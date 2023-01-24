import os

import jax.random as jrandom
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs
from flax.training import train_state

import agent
import config
import environment
import net


def load_model(path, state):
    """Load model from path."""
    params = net.load_params(path)
    state = state.replace(params=params)
    return state


def get_state_object(n_movies_to_recommend: int):
    """Get state object."""
    hparams = config.HParams(n_actions=n_movies_to_recommend)
    dqn_agent = agent.DQNAgent(hparams=hparams)
    state = dqn_agent.init_states()
    state = agent.load_model(path=hparams.checkpoints_dir, state=state)
    return state


def convert_model_to_tensorflow_js_model(state: train_state.TrainState, tfjs_model_dir: str):
    """Convert jax model to tensorflow.js model."""
    tfjs.converters.convert_jax(
        state.apply_fn,
        {'params': state.params},
        # age, gender, occupation, zip_code
        input_signatures=[tf.TensorSpec((1, 4), tf.float32)],
        model_dir=tfjs_model_dir,

    )
