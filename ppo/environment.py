import gymnasium as gym
import jax.numpy as jnp


def create_env(env_name, seed):
    env = gym.make(env_name)
    return env


class JaxEnv:
    def __init__(self, env_name, seed):
        self.env = create_env(env_name, seed)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset()
        return jnp.array(observation), info

    def step(self, action: jnp.ndarray):
        action = action.item()
        observation, reward, done, truncated, info = self.env.step(action)
        return (
            jnp.array(observation),
            jnp.array(reward),
            jnp.array(done),
            jnp.array(truncated),
            info,
        )

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
