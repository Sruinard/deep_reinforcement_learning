import os
import time

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import agent
import config
import environment
import replay_buffer
import checkpoint_to_saved_model

os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')     # tell JAX to use GPU
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'  # don't use all gpu mem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    n_movies_to_recommend = 100
    env = environment.MovieLensEnv(
        data_dir='./data/ml-100k', n_actions=n_movies_to_recommend)
    hparams = config.HParams(n_actions=n_movies_to_recommend)
    dqn_agent = agent.DQNAgent(hparams=hparams)
    memory = replay_buffer.ExperienceReplay(memory_size=hparams.memory_size)
    state = dqn_agent.init_states()
    rng = jrandom.PRNGKey(hparams.seed)
    best_regret = np.inf
    writer = SummaryWriter(comment="-recommender-system-movie-lens")

    for episode_idx in range(hparams.n_train_episodes):
        rng, _ = jrandom.split(rng)
        obs, _ = env.reset()
        is_done = False
        total_reward = 0
        time_start_episode = time.time()
        total_loss = 0.0
        n_steps_in_episode = 1
        total_regret = 0.0

        while not is_done:
            n_steps_in_episode += 1
            obs, a, r, is_done, obs_t = dqn_agent.collect_step(
                state=state,
                env=env,
                observation=obs,
                rng=rng,
            )
            total_regret += r

            memory.push(obs, a, r, is_done, obs_t)
            obs = obs_t

            if memory.is_ready(hparams.batch_size):
                loss, state = dqn_agent.update_step(
                    batch=memory.sample_batch(hparams.batch_size),
                    state=state
                )
                total_loss += loss
            if is_done:
                print(f"Episode {episode_idx} - regret: {total_regret / n_steps_in_episode} - loss: {total_loss / n_steps_in_episode} - steps: {n_steps_in_episode} - time: {time.time() - time_start_episode} - epsilon: {dqn_agent.get_epsilon(state.step)}")
                writer.add_scalar(
                    "Train/loss", float(total_loss / n_steps_in_episode), episode_idx)
                writer.add_scalar("Frames/second", n_steps_in_episode /
                                  (time.time() - time_start_episode), episode_idx)

                # add regret
                writer.add_scalar(
                    "Train/regret", float(total_regret / n_steps_in_episode), episode_idx)

                n_steps_in_episode = 1

        if not episode_idx % hparams.evaluate_every_n_epoch:
            print(f"Episode {episode_idx}")
            total_regret = 0.0
            total_n_steps = 0

            for _ in range(hparams.n_episodes_per_eval):

                n_steps_in_episode = 0
                obs, _ = env.reset()
                epsilon = 0.00  # No exploration
                is_done = False
                while not is_done:
                    total_n_steps += 1
                    a = dqn_agent.policy(
                        state, obs, env.action_space.n, epsilon, rng)
                    next_obs, r, is_done, _, _ = env.step(int(a))
                    total_regret += r
                    obs = next_obs

            avg_regret = total_regret / total_n_steps
            if avg_regret < best_regret:
                print(f"New best regret: {avg_regret} Saving model...")
                best_regret = avg_regret
                agent.save_model(state=state, path=hparams.checkpoints_dir)
                checkpoint_to_saved_model.convert_model_to_tensorflow_js_model(
                    state=state,
                    tfjs_model_dir="./assets/tfjs_model",
                )

            print(f"Average reward: {avg_regret}")
            writer.add_scalar("Eval/average_regret", avg_regret, episode_idx)
            writer.add_scalar("epsilon", float(
                dqn_agent.get_epsilon(state.step)), episode_idx)

    writer.close()


if __name__ == "__main__":
    main()
