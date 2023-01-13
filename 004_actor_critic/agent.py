import net
import rlax
import jax
from flax.training import train_state
import loss_function
import jax.numpy as jnp
import accumulator as acc
import optax
import config


class A2CAgent:

    def __init__(self, a2c_net: net.ActorCritic, train_config: config.TrainConfig) -> None:
        self.a2c_net = a2c_net
        self.train_config = train_config

        self.actor_step = jax.jit(self.actor_step)
        self.update = jax.jit(self.update)

    def init_state(self):
        params = self.a2c_net.init(jax.random.PRNGKey(
            0), jnp.zeros(self.train_config.obs_shape))["params"]
        # optim = optax.adam(learning_rate=self.train_config.learning_rate)
        #
        optim = optax.chain(
            optax.clip_by_global_norm(1.0),  # Clip gradients at norm 1
            optax.adam(learning_rate=self.train_config.learning_rate)
        )
        return train_state.TrainState.create(
            apply_fn=self.a2c_net.apply,
            params=params,
            tx=optim,
        )

    def actor_step(self, state: train_state.TrainState, observation, rng):
        logits, v_t = self.a2c_net.apply(
            {'params': state.params}, observation)
        probs = jax.nn.softmax(logits)
        action = jax.random.choice(key=rng, a=jnp.arange(
            logits.shape[-1]), shape=(1,), p=probs[0])
        return action, logits, v_t

    def update(self, state: train_state.TrainState, trajectory: acc.transition):
        """Compute loss.

        Args:
            trajectory (Trajectory): trajectory
        Returns:
            loss (np.ndarray): loss
        """

        (loss, (critic_loss, policy_loss)), grads = jax.value_and_grad(
            self._loss_fn, has_aux=True)(state.params, trajectory)
        state = state.apply_gradients(grads=grads)
        return state, loss, (critic_loss, policy_loss)

    def _critic_loss(self, params, trajectory: acc.transition, n_steps):
        """Compute critic loss.

        Args:
            y_true (np.ndarray): target values
            y_pred (np.ndarray): predicted values
        Returns:
            loss (np.ndarray): loss
        """
        _, v_tm1 = self.a2c_net.apply({"params": params}, trajectory.obs_tm1)
        _, v_t = self.a2c_net.apply({"params": params}, trajectory.obs_t)

        v_targets = rlax.n_step_bootstrapped_returns(
            r_t=trajectory.r_t,
            discount_t=trajectory.discount_t * (1 - trajectory.done_t),
            v_t=jnp.squeeze(v_t),
            n=n_steps,
            # stop_target_gradients=True
        )
        v_targets = (v_targets - v_targets.mean()) / (v_targets.std() + 1e-8)
        return jnp.mean(rlax.l2_loss(jax.lax.stop_gradient(v_targets) - v_tm1))

    def _policy_loss(self, params, trajectory: acc.transition, n_steps):
        logits_tm1, v_tm1 = self.a2c_net.apply(
            {"params": params}, trajectory.obs_tm1)
        _, v_t = self.a2c_net.apply({"params": params}, trajectory.obs_t)
        v_targets = rlax.n_step_bootstrapped_returns(
            r_t=trajectory.r_t,
            discount_t=trajectory.discount_t * (1 - trajectory.done_t),
            v_t=jnp.squeeze(v_t),
            n=n_steps,
            stop_target_gradients=True
        )
        adv_tm1 = v_targets - jax.lax.stop_gradient(jnp.squeeze(v_tm1))

        adv_tm1 = (adv_tm1 - adv_tm1.mean()) / (adv_tm1.std() + 1e-8)

        policy_loss = rlax.policy_gradient_loss(
            logits_t=logits_tm1,
            a_t=jnp.asarray(trajectory.a_tm1, dtype=jnp.int32),
            adv_t=adv_tm1,
            w_t=jnp.ones_like(adv_tm1)
        )
        return policy_loss

    # def _loss_fn(self, params, trajectory):
    #     critic_loss = self._critic_loss(
    #         params, trajectory, self.train_config.n_steps)
    #     policy_loss = self._policy_loss(
    #         params, trajectory, self.train_config.n_steps)
    #    # entropy = rlax.entropy_loss(logits=trajectory.logits_tm1)
    #     loss = critic_loss + policy_loss  # - entropy * self.train_config.entropy_coeff
    #     return loss, (critic_loss, policy_loss)

    def _loss_fn(self, params, trajectory: acc.transition):

        logits_tm1, v_tm1 = self.a2c_net.apply(
            {"params": params}, trajectory.obs_tm1)
        _, v_t = self.a2c_net.apply({"params": params}, trajectory.obs_t)
        v_targets = rlax.n_step_bootstrapped_returns(
            r_t=trajectory.r_t,
            discount_t=trajectory.discount_t * (1 - trajectory.done_t),
            v_t=jnp.squeeze(v_t),
            n=self.train_config.n_steps,
            stop_target_gradients=True
        )

        v_tm1 = jnp.squeeze(v_tm1)

        # v_targets = rlax.n_step_bootstrapped_returns(
        #     r_t=trajectory.r_t[:-1],
        #     discount_t=trajectory.gamma_t[1:] *
        #     (1 - trajectory.done_t[1:]),
        #     v_t=trajectory.v_t[1:],
        #     n=self.train_config.n_steps,
        #     stop_target_gradients=True
        # )

        # logits, _ = self.a2c_net.apply(
        #     {"params": params}, trajectory.obs_t[:-1])

        v_targets = (v_targets - jnp.mean(v_targets)) / \
            (jnp.std(v_targets) + 1e-8)
        critic_loss = jnp.mean(rlax.l2_loss(
            v_tm1,  jax.lax.stop_gradient(v_targets)))

        assert v_tm1.shape == v_targets.shape
        adv_tm1 = v_targets - jax.lax.stop_gradient(v_tm1)

        policy_loss = rlax.policy_gradient_loss(
            logits_t=logits_tm1,
            a_t=jnp.asarray(trajectory.a_tm1, dtype=jnp.int32),
            adv_t=adv_tm1,
            w_t=jnp.ones_like(adv_tm1)
        )

        # policy_loss = rlax.policy_gradient_loss(
        #     logits_t=logits,
        #     # convert action to int32 as required by rlax
        #     a_t=jnp.asarray(trajectory.a_t[:-1], dtype=jnp.int32),
        #     adv_t=adv_t,
        #     w_t=trajectory.gamma_t[:-1]
        # )

        # beta_t = jnp.ones_like(
        #     trajectory.a_t[:-1]) * self.train_config.entropy_coeff
        # entropy = rlax.entropy_loss(logits, w_t=beta_t)

        # the formula for entropy is -sum(p * log(p)) which is a negative number
        # if we become very confident in our actions, the entropy will be very small negative number
        total_loss = policy_loss + critic_loss  # + entropy
        return total_loss, (critic_loss, policy_loss)  # , entropy, v_targets)
