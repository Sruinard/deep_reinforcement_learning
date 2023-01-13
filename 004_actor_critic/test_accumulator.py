import pytest
import accumulator


def test_accumulator_can_store_transitions():
    obs_tm1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    a_tm1 = [0, 1, 1, 0, 0, 0, 0, 0, 1, 0]
    rewards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    is_done = [False, False, False, False,
               False, False, False, False, False, True]
    obs_t = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    acc = accumulator.TrajectoryAccumulator(trajectory_length=4)

    for obs_tm1, a_tm1, r_t, is_done, obs_t in zip(obs_tm1, a_tm1, rewards, is_done, obs_t):
        acc.push(accumulator.transition(obs_tm1, a_tm1, r_t, is_done, obs_t))

    assert len(acc._trajctory) == 4


def test_accumulator_can_sample():
    obs_tm1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    a_tm1 = [0, 1, 1, 0, 0, 0, 0, 0, 1, 0]
    rewards = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    is_done = [False, False, False, False,
               False, False, False, False, False, True]
    obs_t = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    acc = accumulator.TrajectoryAccumulator(trajectory_length=4)

    for obs_tm1, a_tm1, r_t, is_done, obs_t in zip(obs_tm1, a_tm1, rewards, is_done, obs_t):
        acc.push(accumulator.transition(obs_tm1, a_tm1, r_t, is_done, obs_t))

    trajectory = acc.get_accumulated_trajectory()
    assert trajectory.obs_tm1.shape == (4,)

