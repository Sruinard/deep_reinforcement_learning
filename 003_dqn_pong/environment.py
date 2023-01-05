import gymnasium as gym
import jax
import jax.numpy as jnp
import collections
from gymnasium.wrappers import frame_stack


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs, {}


class AccumulateNSteps(gym.Wrapper):
    def __init__(self, env=gym.Env, repeat_action_n_times=4) -> None:
        super().__init__(env)
        self.repeat_action_n_times = repeat_action_n_times
        self.observation_buffer = collections.deque(maxlen=4)

    def step(self, action):
        reward = 0.0
        done = None
        for _ in range(self.repeat_action_n_times):
            observation, step_reward, done, truncated, info = self.env.step(
                action)
            self.observation_buffer.append(observation)
            reward += step_reward
            if done:
                break
        observation = jnp.max(jnp.stack(self.observation_buffer), axis=0)
        return observation, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        self.observation_buffer.clear()
        observation, info = self.env.reset()
        self.observation_buffer.append(observation)
        return observation, info


class ResizeAndNormalizeFrames(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, transformed_observation_shape=(84, 84, 1)):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=transformed_observation_shape)
        self.transformed_observation_shape = transformed_observation_shape

    def _resize(self, observation):
        resized_obs = jax.image.resize(
            observation, (110, 84, 3), method='nearest')
        cropped_obs = resized_obs[18:102, :].mean(axis=-1, keepdims=True)
        return cropped_obs

    def _normalize(self, observation):
        return observation / 255.0

    def observation(self, observation):
        resized_observation = self._resize(observation)
        normalized_observation = self._normalize(resized_observation)
        return normalized_observation


class AsJaxArray(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        return jnp.asarray(observation)


def create(environment_name: str):
    env_raw = gym.make(environment_name)
    env_press_fire_if_needed_to_start_game = FireResetEnv(env_raw)
    env_skip_n_steps = AccumulateNSteps(env_press_fire_if_needed_to_start_game)
    env_resized_and_normalized = ResizeAndNormalizeFrames(env_skip_n_steps)
    env_stacked_frames = frame_stack.FrameStack(
        env_resized_and_normalized, num_stack=4)
    env_jax_array = AsJaxArray(env_stacked_frames)
    return env_jax_array
