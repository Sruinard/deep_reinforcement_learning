import chex
import jax
import jax.numpy as jnp

def double_q_learning(q_tm1, a_tm1, discount_t, r_t, q_t, q_a, is_done):
    """Double Q-learning objective.
    
    Args:
      q_tm1: Q-values at time t-1.
      a_tm1: Actions at time t-1.
      discount_t: Discount factor at time t.
      r_t: Rewards at time t.
      q_t: Q-values at time t.
      q_a: Selector for Q-values at time t.
      is_done: Whether the episode is done at time t.
      
      Returns:
        TD-error.
    """
    chex.assert_rank([q_tm1, a_tm1, discount_t, r_t, q_t, q_a, is_done], [1,0,0,0,1,0,0])

    target_tm1 = r_t + (1.0 - is_done) * discount_t * q_t[q_a]
    target_tm1 = jax.lax.stop_gradient(target_tm1)
    return target_tm1 - q_tm1[a_tm1]

def double_q_learning_loss(q_tm1, a_tm1, discount_t, r_t, q_t, a_t, is_done):

    batch_double_q_learning = jax.vmap(double_q_learning, in_axes=(0, 0, None, 0, 0, 0, 0))
    td_error = batch_double_q_learning(q_tm1, a_tm1, discount_t, r_t, q_t, a_t, is_done)

    l2_loss = 0.5 * td_error ** 2
    return jnp.mean(l2_loss)




    