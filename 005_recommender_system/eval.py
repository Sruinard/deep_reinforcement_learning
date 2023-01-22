import environment
import net
import agent
import config
import jax.random as jrandom
import numpy as np
import os
from run import FLAGS


env = environment.MovieLensEnv("./data/ml-100k", FLAGS.n_movies_to_recommend)
hparams = config.HParams(n_actions=FLAGS.n_movies_to_recommend)
dqn_agent = agent.DQNAgent(hparams=hparams)
state = dqn_agent.init_states()
state = agent.load_model(path=hparams.checkpoints_dir, state=state)
rng = jrandom.PRNGKey(42)


total_reward = 0
epsilon = 0.00  # No exploration
is_done = False
obs, _ = env.reset()
while not is_done:
    a = dqn_agent.policy(
        state, obs, FLAGS.n_movies_to_recommend, epsilon, rng)
    next_obs, r, is_done, _, _ = env.step(int(a))
    total_reward += r
    obs = next_obs
print(f"Total reward: {total_reward}")
