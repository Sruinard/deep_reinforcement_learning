
import jax
import collections
import jax.numpy as jnp

transition = collections.namedtuple(
    'transition', ['obs_tm1', 'a_tm1', 'r_t', 'done_t', 'discount_t', 'obs_t'])


class TrajectoryAccumulator:

    """ Accumulates trajectories and returns them as batches. """

    def __init__(self, trajectory_length: int) -> None:
        self._trajectory = collections.deque(maxlen=trajectory_length)

    def push(self, transition: collections.namedtuple) -> None:
        self._trajectory.append(transition)

    def get_accumulated_trajectory(self) -> transition:
        return jax.tree_map(
            lambda *transitions: jnp.stack(transitions, dtype=jnp.float32),
            *self._trajectory
        )

    def is_ready(self):
        return len(self._trajectory) == self._trajectory.maxlen
