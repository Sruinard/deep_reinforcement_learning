
import time
from typing import List

import jax
import jax.numpy as jnp
import numpy as np


def add_grads_to_summary(grads, writer, episode_idx):
    [
        writer.add_histogram(f"grads/{i}/kernel", np.asarray(layer_grads), episode_idx) 
        if 
            i % 2 == 0 
        else 
            writer.add_histogram(f"grads/{i}/bias", np.asarray(layer_grads), episode_idx) 
        for i, layer_grads in enumerate(jax.tree_util.tree_flatten(grads)[0])
    ]