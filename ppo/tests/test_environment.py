import jax.numpy as jnp
import environment


def test_environment_can_reset_and_take_multiple_steps():
    env = environment.JaxEnv("CartPole-v1", 0)
    observation, info = env.reset()
    assert observation.shape == (4,)
    assert info == {}
    for _ in range(10):
        action = jnp.array(0)
        observation, reward, done, truncated, info = env.step(action)
        assert observation.shape == (4,)
        assert reward.shape == ()
        assert done.shape == ()
        assert truncated.shape == ()
        assert info == {}
    env.close()
