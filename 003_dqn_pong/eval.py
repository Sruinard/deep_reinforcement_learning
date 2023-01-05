import environment
import net
import agent
import config
import jax.random as jrandom
import numpy as np


env = environment.create("PongNoFrameskip-v4")
hparams = config.HParams(n_actions=env.action_space.n)
dqn_agent = agent.DQNAgent(hparams=hparams)
state = dqn_agent.init_states()
state = agent.load_model(path=hparams.checkpoints_dir, state=state)
rng = jrandom.PRNGKey(42)


do_save_agent_playing = True
total_reward = 0
epsilon = 0.00  # No exploration
is_done = False
obs, _ = env.reset()
frames = []
while not is_done:
    a = dqn_agent.policy(
        state, obs, env.action_space.n, 0.00, rng)
    next_obs, r, is_done, _, _ = env.step(a)
    total_reward += r
    obs = next_obs
    frames.append(np.asarray(obs[0]))
print(f"Total reward: {total_reward}")

if do_save_agent_playing:
    from moviepy.editor import ImageSequenceClip
    clip = ImageSequenceClip(list(np.asarray(frames) * 255.0), fps=25)
    clip.write_gif("pong.gif", fps=25)
