import config
import net
import accumulator as acc
import gymnasium as gym
import agent
import jax
import jax.numpy as jnp


def run_loop(agent: agent.A2CAgent, env: gym.Env, accumulator: acc.TrajectoryAccumulator, train_config: config.TrainConfig):
    rng = jax.random.PRNGKey(train_config.seed)
    state = agent.init_state()

    rewards_history = []

    for train_episode in range(train_config.n_train_episodes):
        train_reward = 0
        obs_tm1, _ = env.reset()
        done_t = False
        while not done_t:
            rng, action_rng = jax.random.split(rng)
            a_tm1, logits_tm1, v_tm1 = agent.actor_step(
                state, obs_tm1[jnp.newaxis, :], action_rng)
            obs_t, r_t, done_t, _, _ = env.step(int(a_tm1))
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
            train_reward += r_t

            obs_tm1 = obs_t

            if done_t:  # accumulator.is_ready():
                trajectory = accumulator.get_accumulated_trajectory()
                state, loss, (critic_loss, policy_loss) = agent.update(
                    state, trajectory)
                accumulator._trajectory.clear()
                rewards_history.append(train_reward)
                reward_last_100 = jnp.asarray(rewards_history[-100:]).mean()
                print(
                    f"Episode {train_episode} | Train reward: {train_reward} | Last 100 reward: {reward_last_100}")
                print(
                    f"Loss: {loss} | Critic loss: {critic_loss} | Policy loss: {policy_loss}")

        if reward_last_100 > 195.0:
            print(f"Environment solved in {train_episode} episodes")
            break

        if train_episode % train_config.eval_every_n_episodes == 0:
            average_eval_reward = 0
            for eval_episode in range(train_config.n_eval_episodes):
                episode_reward = 0
                obs_tm1, _ = env.reset()
                done = False
                while not done:
                    a_tm1, _, _ = agent.actor_step(
                        state, obs_tm1[jnp.newaxis, :], rng, is_deterministic=True)
                    obs_t, r_t, done_t, _, _ = env.step(int(a_tm1))
                    obs_tm1 = obs_t
                    episode_reward += r_t
                    if done_t:
                        average_eval_reward += episode_reward
                        break
            average_eval_reward /= train_config.n_eval_episodes
            print(
                f"Episode {train_episode} | Average eval reward: {average_eval_reward}")


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
