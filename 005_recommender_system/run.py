import os
import time

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from absl import flags, app

import agent
import config
import environment
import replay_buffer
import checkpoint_to_saved_model

os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')     # tell JAX to use GPU
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'  # don't use all gpu mem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# flags n movies to recommend
flags.DEFINE_integer("n_movies_to_recommend", 100,
                     "Number of movies to recommend")

# parse flags
FLAGS = flags.FLAGS


def main(argv):
    print(f"Learning to recommend {FLAGS.n_movies_to_recommend} movies...")
    # check for gpu
    print("Found", jax.local_device_count(), "GPUs")
    hparams = config.HParams(n_actions=FLAGS.n_movies_to_recommend)
    rng = jrandom.PRNGKey(hparams.seed)
    env = environment.MovieLensEnv(
        data_dir='./data/ml-100k', n_actions=FLAGS.n_movies_to_recommend, rng=rng)
    dqn_agent = agent.DQNAgent(hparams=hparams)
    memory = replay_buffer.ExperienceReplay(memory_size=hparams.memory_size)
    state = dqn_agent.init_states()
    best_reward = -np.inf
    writer = SummaryWriter(comment="-recommender-system-movie-lens")

    for episode_idx in range(hparams.n_train_episodes):
        rng, episode_rng = jrandom.split(rng)
        obs, _ = env.reset(rng=episode_rng)
        is_done = False
        total_reward = 0
        time_start_episode = time.time()
        total_loss = 0.0
        n_steps_in_episode = 1

        while not is_done:
            rng, action_rng = jrandom.split(rng)
            n_steps_in_episode += 1
            obs, a, r, is_done, obs_t = dqn_agent.collect_step(
                state=state,
                env=env,
                observation=obs,
                rng=action_rng,
            )
            total_reward += r

            memory.push(obs, a, r, is_done, obs_t)
            obs = obs_t

            if memory.is_ready(hparams.memory_size):
                batch = memory.sample_batch(hparams.batch_size)
                loss, state = dqn_agent.update_step(
                    batch=batch,
                    state=state
                )

                total_loss += loss

                if is_done:
                    print(f"Episode {episode_idx} - Reward: {total_reward / n_steps_in_episode} - loss: {total_loss / n_steps_in_episode} - steps: {n_steps_in_episode} - time: {time.time() - time_start_episode} - epsilon: {dqn_agent.get_epsilon(state.step)}")
                    writer.add_scalar(
                        "Train/loss", float(total_loss / n_steps_in_episode), episode_idx)
                    writer.add_scalar("Frames/second", n_steps_in_episode /
                                      (time.time() - time_start_episode), episode_idx)

                    writer.add_scalar(
                        "Train/reward", float(total_reward / n_steps_in_episode), episode_idx)

                    n_steps_in_episode = 1

        if not episode_idx % hparams.evaluate_every_n_epoch:
            print(f"Episode {episode_idx}")
            total_reward = 0.0
            total_n_steps = 0

            for _ in range(hparams.n_episodes_per_eval):

                n_steps_in_episode = 0
                rng, episode_rng = jrandom.split(rng)
                obs, _ = env.reset(rng=episode_rng)
                epsilon = 0.00  # No exploration
                is_done = False
                while not is_done:
                    rng, action_rng, step_rng = jrandom.split(rng, 3)
                    total_n_steps += 1
                    a = dqn_agent.policy(
                        state, obs, env.action_space.n, epsilon, action_rng)
                    next_obs, r, is_done, _, _ = env.step(int(a), rng=step_rng)
                    total_reward += r
                    obs = next_obs

            avg_reward = total_reward / total_n_steps
            if avg_reward > best_reward:
                print(f"New best reward: {avg_reward} Saving model...")
                best_reward = avg_reward
                agent.save_model(state=state, path=hparams.checkpoints_dir)
                checkpoint_to_saved_model.convert_model_to_tensorflow_js_model(
                    state=state,
                    tfjs_model_dir="./assets/tfjs_model",
                )

            print(f"Average reward: {avg_reward}")
            writer.add_scalar("Eval/average_reward",
                              float(avg_reward), episode_idx)
            writer.add_scalar("epsilon", float(
                dqn_agent.get_epsilon(state.step)), episode_idx)

    writer.close()


if __name__ == "__main__":
    app.run(main)
