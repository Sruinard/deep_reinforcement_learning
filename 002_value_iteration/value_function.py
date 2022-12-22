import gymnasium as gym

class Agent:

    def compute_state_value(self, env: gym.Environment):


        total_n_transitions = 0
        for action in env.actions:
            n_times_visited_state_prime = 0
            self.transition_probability(total_n_transitions, n_times_visited_state_prime)