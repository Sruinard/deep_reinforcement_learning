import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import collections


class Agent:

    def __init__(self, env: gym.Env):
        self.env = env
        self.rewards = collections.defaultdict(float)
        self.transition = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    def interact_using_random_policy(self, n_steps: int):
        state, *_ = self.env.reset()
        for _ in range(n_steps):
            action = self.env.action_space.sample()
            next_state, reward, is_done, *_ = self.env.step(action=action)
            self.rewards[(state, action, next_state)] = reward
            self.transition[(state, action)][next_state] += 1
            state, _ = self.env.reset() if is_done else (next_state, {})

    def compute_value_of_taking_action_a_in_state_s(self, state, action, gamma=0.9):
        next_states = self.transition[(state, action)]
        total_n_transitions_from_state_s = sum(next_states.values())
        action_value = 0.0
        for next_state, n_times_visited_next_state in next_states.items():
            # reward for taking action in given state
            # differnt from value iteration. reward is a function of state and action.
            reward = self.rewards[(state, action, next_state)]
            # differnt from value iteration. select best action in next state
            best_action = self.act_greedy(next_state)
            action_value += (n_times_visited_next_state / total_n_transitions_from_state_s) * (
                reward + gamma * self.values[next_state, best_action])  # transition_dynamics * (reward + discount_factor * value_of_next_q_value)

        return action_value

    def compute_q_function(self, gamma=0.9):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = self.compute_value_of_taking_action_a_in_state_s(
                    state, action, gamma)
                # act optimally by selecting maximum value of all actions in given state
                # differnt from value iteration. select best action in next state
                self.values[state, action] = action_value

    def act_greedy(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            # different from value iteration. Here we use the q function.
            action_value = self.values[state, action]
            if best_action is None or action_value > best_value:
                best_value = action_value
                best_action = action
        return best_action

    def evaluate_policy(self, env: gym.Env):
        state, _ = env.reset()
        total_reward = 0.0
        while True:
            action = self.act_greedy(state)
            state, reward, is_done, *_ = env.step(action)
            total_reward += reward
            if is_done:
                break
        return total_reward

    def train_step(self, n_steps: int):
        self.interact_using_random_policy(n_steps)
        self.compute_q_function()

    def eval(self, n_episodes: int,  env: gym.Env):
        reward = 0.0
        for _ in range(n_episodes):
            reward += self.evaluate_policy(env)
        return reward / n_episodes  # average reward


class TrainConfig:

    def __init__(self, n_steps: int, n_episodes: int, gamma: float):
        self.n_steps = n_steps
        self.n_episodes = n_episodes
        self.gamma = gamma


if __name__ == '__main__':
    env_train = gym.make('FrozenLake-v1')
    env_eval = gym.make('FrozenLake-v1')
    train_config = TrainConfig(n_steps=100, n_episodes=20, gamma=0.9)
    agent = Agent(env_train)
    writer = SummaryWriter(comment="-q-learning")

    iter_idx = 0
    while True:
        iter_idx += 1
        agent.train_step(n_steps=train_config.n_steps)
        reward = agent.eval(n_episodes=train_config.n_episodes, env=env_eval)
        writer.add_scalar('reward', reward, iter_idx)

        if reward > 0.8:
            print(f'Finished training after {iter_idx} iterations')
            break

        if iter_idx % 100 == 0:
            print(f'Iteration {iter_idx}, reward {reward}')
    writer.close()
