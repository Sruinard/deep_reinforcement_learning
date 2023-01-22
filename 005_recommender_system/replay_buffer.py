import collections
import random

import jax.numpy as jnp


class ExperienceReplay:

    def __init__(self, memory_size: int):
        self.memory = collections.deque(maxlen=memory_size)

    def push(self, observation, action, reward, is_done, next_observation):
        experience = (observation, action, reward, is_done, next_observation)
        self.memory.append(experience)

    def sample_batch(self, batch_size: int):
        batch = random.sample(self.memory, batch_size)
        obs_tm1, a_tm1, r_t, is_done, obs_t = zip(*batch)
        return (
            jnp.asarray(obs_tm1),
            jnp.asarray(a_tm1),
            jnp.asarray(r_t),
            jnp.asarray(is_done),
            jnp.asarray(obs_t),
        )
    
    def is_ready(self, memory_size: int):
        return len(self.memory) >= memory_size
    