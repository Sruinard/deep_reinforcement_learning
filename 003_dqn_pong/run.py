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
import metrics
import replay_buffer

os.environ.setdefault('JAX_PLATFORM_NAME', 'gpu')     # tell JAX to use GPU
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'  # don't use all gpu mem
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

REWARD_BOUND_TO_SOLVE_PONG = 19.5


def main():

    env = environment.create("PongNoFrameskip-v4")
    hparams = config.HParams(n_actions=env.action_space.n)
    dqn_agent = agent.DQNAgent(hparams=hparams)
    memory = replay_buffer.ExperienceReplay(memory_size=hparams.memory_size)
    state = dqn_agent.init_states()
    rng = jrandom.PRNGKey(hparams.seed)
    best_reward = -np.inf
    writer = SummaryWriter(log_dir=hparams.log_dir, comment="-dqn-pong")

    for episode_idx in range(hparams.n_train_episodes):
        rng, _ = jrandom.split(rng)
        obs, _ = env.reset()
        is_done = False
        total_reward = 0
        time_start_episode = time.time()
        total_loss = 0.0
        n_steps_in_episode = 1

        while not is_done:
            n_steps_in_episode += 1
            obs, a, r, is_done, obs_t = dqn_agent.collect_step(
                state=state,
                env=env,
                observation=obs,
                rng=rng,
            )

            memory.push(obs, a, r, is_done, obs_t)
            obs = obs_t

            if memory.is_ready(hparams.batch_size):
                (loss, grads), state = dqn_agent.update_step(
                    batch=memory.sample_batch(hparams.batch_size),
                    state=state,
                    discount_t=hparams.gamma
                )
                total_loss += loss
            if is_done:
                metrics.add_grads_to_summary(grads, writer, episode_idx)
                writer.add_scalar(
                    "Train/loss", float(total_loss / n_steps_in_episode), episode_idx)
                writer.add_scalar("Frames/second", n_steps_in_episode /
                                  (time.time() - time_start_episode), episode_idx)
                n_steps_in_episode = 1

            if not state.step % hparams.sync_every_n_steps:
                state = dqn_agent.sync_target_network(state)

        if not episode_idx % hparams.evaluate_every_n_epoch:
            print(f"Episode {episode_idx}")
            total_reward = 0

            for _ in range(hparams.n_episodes_per_eval):
                obs, _ = env.reset()
                epsilon = 0.00  # No exploration
                is_done = False
                while not is_done:
                    a = dqn_agent.policy(
                        state, obs, env.action_space.n, epsilon, rng)
                    next_obs, r, is_done, _, _ = env.step(a)
                    total_reward += r
                    obs = next_obs

            avg_reward = total_reward / hparams.n_episodes_per_eval
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save_model(state=state, path=hparams.checkpoints_dir)
            print(f"Average reward: {avg_reward}")
            writer.add_scalar("Eval/average_reward", avg_reward, episode_idx)
            writer.add_scalar("epsilon", float(
                dqn_agent.get_epsilon(state.step)), episode_idx)

            if avg_reward > REWARD_BOUND_TO_SOLVE_PONG:
                print(f"Solved in {episode_idx} episodes!")
                break
    writer.close()


if __name__ == "__main__":
    main()
