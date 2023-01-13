import net
import rlax
import jax
from flax.training import train_state
import jax.numpy as jnp
import accumulator as acc
import optax
import config


class A2CAgent:

    def __init__(self, a2c_net: net.ActorCritic, train_config: config.TrainConfig) -> None:
        self.a2c_net = a2c_net
        self.train_config = train_config

        # jit functions for speed
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

    def actor_step(self, state: train_state.TrainState, observation, rng, is_deterministic=False):
        logits, v_t = self.a2c_net.apply(
            {'params': state.params}, observation)
        probs = jax.nn.softmax(logits)
        action = jax.lax.select(
            is_deterministic,
            jax.random.choice(key=rng, a=jnp.arange(
                logits.shape[-1]), shape=(1,), p=probs[0]),
            jnp.argmax(logits, axis=-1)
        )
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

    def _loss_fn(self, params, trajectory: acc.transition):

        logits_tm1, v_tm1 = self.a2c_net.apply(
            {"params": params}, trajectory.obs_tm1)
        _, v_t = self.a2c_net.apply({"params": params}, trajectory.obs_t)

        # n-step bootstrapped returns
        v_targets = rlax.n_step_bootstrapped_returns(
            r_t=trajectory.r_t,
            discount_t=trajectory.discount_t * (1 - trajectory.done_t),
            v_t=jnp.squeeze(v_t),
            n=self.train_config.n_steps,
            stop_target_gradients=True
        )
        v_tm1 = jnp.squeeze(v_tm1)

        # normalize targets
        v_targets = (v_targets - jnp.mean(v_targets)) / \
            (jnp.std(v_targets) + 1e-8)

        # compute critic loss
        critic_loss = jnp.mean(rlax.l2_loss(
            v_tm1,  jax.lax.stop_gradient(v_targets)))

        assert v_tm1.shape == v_targets.shape

        # compute advantage function (make sure gradients are not propagated to critic)
        adv_tm1 = v_targets - jax.lax.stop_gradient(v_tm1)

        # compute policy loss (for the actor)
        policy_loss = rlax.policy_gradient_loss(
            logits_t=logits_tm1,
            a_t=jnp.asarray(trajectory.a_tm1, dtype=jnp.int32),
            adv_t=adv_tm1,
            w_t=jnp.ones_like(adv_tm1)
        )

        # beta_t = jnp.ones_like(
        #     trajectory.a_t[:-1]) * self.train_config.entropy_coeff
        # entropy = rlax.entropy_loss(logits, w_t=beta_t)

        # the formula for entropy is -sum(p * log(p)) which is a negative number
        # if we become very confident in our actions, the entropy will be very small negative number
        total_loss = policy_loss + critic_loss  # + entropy
        return total_loss, (critic_loss, policy_loss)  # , entropy, v_targets)
