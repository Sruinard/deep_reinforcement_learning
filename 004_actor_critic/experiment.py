import gymnasium as gym
import jax
import jax.numpy as jnp
from torch.utils.tensorboard import SummaryWriter

import accumulator as acc
import agent
import config
import net
import logging

logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)


def play_episode(agent: agent.A2CAgent, env: gym.Env, accumulator: acc.TrajectoryAccumulator, state: agent.train_state.TrainState, rng: jax.random.PRNGKey, is_eval: bool = False):
    total_episode_reward = 0
    obs_tm1, _ = env.reset()
    done_t = False
    while not done_t:
        rng, action_rng = jax.random.split(rng)
        a_tm1, logits_tm1, v_tm1 = agent.actor_step(
            state, obs_tm1[jnp.newaxis, :], action_rng, is_deterministic=is_eval)
        obs_t, r_t, done_t, _, _ = env.step(int(a_tm1))

        if not is_eval:
            accumulator.push(
                acc.transition(
                    obs_tm1=obs_tm1,
                    a_tm1=a_tm1[0],
                    r_t=r_t,
                    done_t=done_t,
                    obs_t=obs_t,
                    discount_t=train_config.gamma  # scalar
                )
            )
        obs_tm1 = obs_t
        total_episode_reward += r_t
    return total_episode_reward, rng


def update(agent: agent.A2CAgent, state: agent.train_state.TrainState, accumulator: acc.TrajectoryAccumulator, train_config: config.TrainConfig):
    trajectory = accumulator.get_accumulated_trajectory()
    state, loss, (critic_loss, policy_loss) = agent.update(state, trajectory)
    accumulator.clear()
    return state, loss, (critic_loss, policy_loss)


def run_loop(agent: agent.A2CAgent, env: gym.Env, accumulator: acc.TrajectoryAccumulator, train_config: config.TrainConfig):
    rng = jax.random.PRNGKey(train_config.seed)
    state = agent.init_state()
    writer = SummaryWriter(log_dir=train_config.log_dir, comment="-a2c")

    rewards_history = []
    logging.info("Starting training loop...")
    for train_episode in range(train_config.n_train_episodes):
        episode_reward, rng = play_episode(agent, env, accumulator, state, rng)
        state, loss, (critic_loss, policy_loss) = update(
            agent, state, accumulator, train_config)

        rewards_history.append(episode_reward)
        running_train_reward = jnp.asarray(rewards_history[-100:]).mean()

        # evaluation
        if train_episode % train_config.eval_every_n_episodes == 0:
            average_eval_reward = 0.0
            for _ in range(train_config.n_eval_episodes):
                eval_reward, rng = play_episode(
                    agent, env, accumulator, state, rng, is_eval=True)
                average_eval_reward += eval_reward

            average_eval_reward /= train_config.n_eval_episodes

            # logging and writing to tensorboard
            logging.info(
                f"Epoch {train_episode} | Average Train Reward last 100 episodes {running_train_reward} | Train reward: {episode_reward} | Average Eval reward: {average_eval_reward} | Loss: {loss} | Critic loss: {critic_loss} | Policy loss: {policy_loss}")
            [writer.add_scalar(tag, float(value), train_episode) for tag, value in zip(
                ["train/loss", "train/critic_loss", "train/policy_loss", "train/reward", "eval/reward"], [loss, critic_loss, policy_loss, episode_reward, average_eval_reward])]

        if running_train_reward > 195:
            print("Solved!")
            break


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    train_config = config.TrainConfig(
        obs_shape=(1, ) + env.observation_space.shape,
        n_actions=env.action_space.n,
        n_train_episodes=10000,
        n_eval_episodes=1
    )
    a2c_agent = agent.A2CAgent(
        a2c_net=net.ActorCritic(
            train_config.n_actions
        ),
        train_config=train_config
    )
    accumulator = acc.TrajectoryAccumulator(
        trajectory_length=train_config.max_trajectory_length
    )

    run_loop(a2c_agent, env, accumulator, train_config)
