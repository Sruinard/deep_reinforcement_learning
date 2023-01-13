import rlax
import accumulator
import jax.numpy as jnp
from flax.training import train_state
import jax


def n_step_target_from_trajectory(trajectory: accumulator.transition, n_steps: int):
    """Compute n-step target values from trajectory.

    Args:
        trajectory (Trajectory): trajectory
    Returns:
        target_values (np.ndarray): target values
    """

    r_t = trajectory.r_t[1:]
    gamma_t = trajectory.discount_t * (1 - trajectory.done_t)
    v_t = trajectory.v_t[1:]
    targets = rlax.n_step_bootstrapped_returns(
        r_t=r_t,
        discount_t=gamma_t,
        v_t=v_t,
        n=n_steps
    )
    return targets


def update(self, state: train_state.TrainState, trajectory: accumulator.transition):
    """Compute loss.

    Args:
        trajectory (Trajectory): trajectory
    Returns:
        loss (np.ndarray): loss
    """

    (loss, (critic_loss, policy_loss, entropy, v_targets)), grads = jax.value_and_grad(
        self._loss_fn, has_aux=True)(state.params, trajectory)
    state = state.apply_gradients(grads=grads)
    return state, loss, (critic_loss, policy_loss, entropy, v_targets)



def actor_loss(logits_t, actions_t, advantage_t, gamma_t):
    """Compute actor loss.

    Args:
        trajectory (Trajectory): trajectory
    Returns:
        loss (np.ndarray): loss
    """

    policy_loss = rlax.policy_gradient_loss(
        logits_t=logits_t, a_t=actions_t, adv_t=advantage_t, w_t=gamma_t)

    return policy_loss


def entropy_loss(logits_t, beta_t):
    return rlax.entropy_loss(logits_t, beta_t)


def loss_fn(trajectory, n_steps, entropy_coeff):
    """Compute loss.

    Args:
        trajectory (Trajectory): trajectory
    Returns:
        loss (np.ndarray): loss
    """

    target_values = n_step_target_from_trajectory(trajectory, n_steps=n_steps)
    critic_value_loss = critic_loss(target_values, trajectory.v_t[:-1])
    policy_loss = actor_loss(
        logits_t=trajectory.logits_t[:-1],
        actions_t=trajectory.a_tm1[:-1],
        advantage_t=target_values - trajectory.v_t[:-1],
        gamma_t=trajectory.gamma_t[:-1]
    )
    beta_t = jnp.ones_like(trajectory.a_tm1[:-1]) * entropy_coeff
    entropy_loss = entropy_loss(
        trajectory.logits_t[:-1], beta_t=beta_t)

    # subtract entropy loss to encourage exploration
    return critic_value_loss + policy_loss - entropy_loss


def update(state: train_state.TrainState, trajectory, n_steps, entropy_coeff):
    """Compute loss.

    Args:
        trajectory (Trajectory): trajectory
    Returns:
        loss (np.ndarray): loss
    """

    def _loss_fn(params):
        v_targets = rlax.n_step_bootstrapped_returns(
            r_t=trajectory.r_t[1:],
            discount_t=trajectory.gamma_t[1:] * (1 - trajectory.done_t[1:]),
            v_t=trajectory.v_t[1:],
            n=n_steps
        )
        critic_loss = jnp.mean(rlax.l2_loss(trajectory.v_t[:-1], v_targets))

        logits, _ = state.apply_fn({"params": params}, trajectory.obs_tm1[:-1])
        adv_t = v_targets - trajectory.v_t[:-1]

        policy_loss = rlax.policy_gradient_loss(
            logits_t=logits,
            a_t=trajectory.a_tm1[:-1],
            adv_t=adv_t,
            w_t=trajectory.gamma_t[:-1]
        )

        beta_t = jnp.ones_like(trajectory.a_tm1[:-1]) * entropy_coeff
        entropy = rlax.entropy_loss(logits, beta_t=beta_t)

        # the formula for entropy is -sum(p * log(p)) which is a negative number
        # if we become very confident in our actions, the entropy will be very small negative number
        total_loss = policy_loss + critic_loss + entropy
        return total_loss

    loss, grads = jax.value_and_grad(_loss_fn)(state.params, trajectory)
    state = state.apply_gradients(grads=grads)
    return state, loss
