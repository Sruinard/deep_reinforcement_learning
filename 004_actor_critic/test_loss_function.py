import pytest
import jax
import jax.random as jrandom
import jax.numpy as jnp

import accumulator
import loss_function
import net
import rlax


@pytest.fixture
def initialized_model():
    n_actions = 2
    model = net.ActorCritic(n_actions=n_actions)
    rng = jrandom.PRNGKey(0)
    obs = jnp.array([1.0, 2.0, 3.0, 4.0])
    params = model.init(rng, jnp.zeros(obs[jnp.newaxis, :].shape))["params"]
    return model, params


@pytest.fixture
def accumulator_with_trajectory():
    n_actions = 2
    trajectory_length = 5
    acc = accumulator.TrajectoryAccumulator(
        trajectory_length=trajectory_length)

    rng = jrandom.PRNGKey(0)
    obs = jnp.array([1.0, 2.0, 3.0, 4.0])
    model = net.ActorCritic(n_actions=n_actions)
    params = model.init(rng, jnp.zeros(obs[jnp.newaxis, :].shape))["params"]
    gamma = 0.9
    for reward in range(trajectory_length):
        a_logits, v_t = model.apply({"params": params}, obs[jnp.newaxis, :])
        v_t = v_t[0][0]  # remove batch dimension and turn into scalar
        a_tm1 = jnp.argmax(a_logits, axis=-1)[0]
        next_obs = obs + 1.0

        if reward == trajectory_length - 1:
            # last step
            acc.push(accumulator.transition(
                obs, a_tm1, reward, True, v_t, gamma))
        else:
            # not last step
            acc.push(accumulator.transition(
                obs, a_tm1, reward, False, v_t, gamma))
        obs = next_obs
    return acc


@pytest.fixture
def trajectory() -> accumulator.transition:
    trajectory = accumulator.transition(
        obs_tm1=jnp.array([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
            [5.0, 6.0, 7.0, 8.0],
        ]),
        a_tm1=jnp.array([0, 1, 0, 1, 0]),
        r_t=jnp.array([2.0, 2.0, 2.0, 2.0, 2.0]),
        done_t=jnp.array([False, False, False, False, True]),
        discount_t=jnp.array([0.9, 0.9, 0.9, 0.9, 0.9]),
        obs_t=jnp.array([
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
            [5.0, 6.0, 7.0, 8.0],
            [6.0, 7.0, 8.0, 9.0]
        ]),
    )
    # v_tm1 = jnp.array([1.5, 2.5, 3.5, 4.5, 5.5]),
    # v_t = jnp.array([1.5, 2.5, 3.5, 4.5, 5.5]),

    return trajectory

def test_n_step_targets_from_trajectory(trajectory):
    n_steps = 3

    v_tm1 = jnp.array([1.5, 2.5, 3.5, 4.5, 5.5])
    v_t = jnp.array([2.5, 3.5, 4.5, 5.5, 6.5])

    v_targets = rlax.n_step_bootstrapped_returns(
        r_t=trajectory.r_t,
        discount_t=trajectory.discount_t * (1.0 - trajectory.done_t),
        v_t=v_t,
        n=n_steps,
        stop_target_gradients=True
    )



    # target_values = loss_function.n_step_target_from_trajectory(
    #     trajectory=trajectory, n_steps=n_steps)
    # assert target_values.shape == (5,)

    # r_t + gamma_t * r_tp1 + gamma_t**2 * r_tp2 + gamma_t**3 * v_tp3
    expected_first_target_value = 2.0 + 0.9 * 2.0 + 0.9**2 * 2.0 + 0.9**3 * 4.5
    2.0 + 0.9 * 2.0 + 0.9 ** 2 * 2.0

    # r_Tm1 + gamma_T * v_T
    expected_last_target_value = 2.0 + 0.9 * (1.0 - True) * 6.5
    assert jnp.all(v_targets[jnp.array([0, -1])] == jnp.asarray(
        [expected_first_target_value, expected_last_target_value]))


def test_critic_loss(trajectory):
    n_steps = 3

    target_values = loss_function.n_step_target_from_trajectory(
        trajectory=trajectory, n_steps=n_steps)
    assert target_values.shape == (4,)

    # r_t + gamma_t * r_tp1 + gamma_t**2 * r_tp2 + gamma_t**3 * v_tp3
    expected_first_target_value = 2.0 + 0.9 * 2.0 + 0.9**2 * 2.0 + 0.9**3 * 4.5

    # r_Tm1 + gamma_T * v_T
    expected_last_target_value = 2.0 + 0.0 * 5.5
    assert jnp.all(target_values[jnp.array([0, -1])] == jnp.asarray(
        [expected_first_target_value, expected_last_target_value]))

    expected_critical_value_loss = 8.339199
    critic_value_loss = loss_function.critic_loss(
        y_true=target_values, y_pred=trajectory.v_t[:-1])
    assert critic_value_loss == expected_critical_value_loss


def test_actor_loss(trajectory, initialized_model):
    model, params = initialized_model
    target_values = loss_function.n_step_target_from_trajectory(
        trajectory=trajectory, n_steps=3)

    advantage_t = target_values - trajectory.v_t[:-1]

    logits_t, _ = model.apply({"params": params}, trajectory.obs_tm1)

    logits_t = logits_t[:-1]
    actions_t = trajectory.a_tm1[:-1]
    gamma_t = trajectory.gamma_t[:-1]

    policy_loss = loss_function.actor_loss(
        logits_t=logits_t,
        actions_t=actions_t,
        advantage_t=advantage_t,
        gamma_t=gamma_t
    )
    assert policy_loss == 0.39274716


def test_entropy_loss():
    entropy_loss = loss_function.entropy_loss(
        jnp.array([0.5, 0.5]), jnp.array([0.01, 0.01]))
