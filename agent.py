import numpy as np

from keras.utils import to_categorical
from environment import Environment
from misc_functions import Functions
from percept import Percept
from strategies.deepqlearning import DeepQLearning
from strategies.learningstrategy import LearningStrategy


class Agent:
    def __init__(self, environment: Environment, strategy: LearningStrategy):
        self.environment = environment
        self.strategy = strategy

    def learn(self, n_episodes: int):
        episode_count = 0

        # stats variables:
        finished_count_100eps = 0
        x_vals = []
        y_vals = []
        scores = []

        if 'CartPole' in self.environment.env.unwrapped.spec.id:
            while episode_count < n_episodes:
                current_state = self.environment.reset()
                # self.environment.render()
                score_count = 0
                percept = Percept(0, 0, 0, 0, False)
                while not percept.done:
                    # self.environment.render()
                    action = self.strategy.next_action(current_state)
                    next_state, reward, done = self.environment.step(action)
                    percept = Percept(current_state, action, next_state, reward, done)
                    self.strategy.learn(percept, episode_count)
                    current_state = percept.next_state
                    score_count += percept.reward

                scores.append(score_count)
                print("Agent died, got score of: " + str(score_count))
                if episode_count > 0 and episode_count % 10 == 0:
                    print("\nEpisode: " + str(episode_count))
                    print("Exploration probability: " + str(self.strategy.expl_prob))
                    print("Average score:" + str(np.mean(scores)))
                    scores.clear()

                episode_count += 1
        else:
            while episode_count < n_episodes:
                current_state = self.environment.reset()
                percept = Percept(0, 0, 0, 0, False)

                percept_count = 0

                while not percept.done:
                    percept_count += 1

                    action = self.strategy.next_action(current_state)
                    next_state, reward, done = self.environment.step(action)
                    percept = Percept(current_state, action, next_state, reward, done)
                    self.strategy.learn(percept, episode_count)
                    current_state = percept.next_state

                    if percept.reward > 0:
                        finished_count_100eps += 1

                if episode_count > 0 and episode_count % 100 == 0:
                    self.print_stats_last_episodes(episode_count, finished_count_100eps)
                    x_vals.append(episode_count)
                    y_vals.append(finished_count_100eps)
                    finished_count_100eps = 0

                episode_count += 1

            self.print_policy()

            Functions.plot_success_rate(x_vals, y_vals)

    def print_stats_last_episodes(self, episode_count, finished_count_100eps):
        print("\n\nEpisode: " + str(episode_count))
        print("Exploration probability: " + str(self.strategy.expl_prob))
        print("Policy:\n")
        if isinstance(self.strategy, DeepQLearning):
            for state in range(self.environment.observation_space_size):
                print(self.strategy.model_1.predict(
                    to_categorical(state, num_classes=self.environment.observation_space_size).reshape(
                        (1, self.environment.observation_space_size))).squeeze())
        else:
            print(self.strategy.policy)
        print("Amount of times the goal was reached last 100 episodes:\n" + str(finished_count_100eps))

    def print_policy(self):
        if isinstance(self.strategy, DeepQLearning):
            q_table = np.empty([self.environment.observation_space_size, self.environment.action_space_size])
            for state in range(self.environment.observation_space_size):
                q_table[state] = self.strategy.model_1.predict(
                    to_categorical(state, num_classes=self.environment.observation_space_size).reshape(
                        (1, self.environment.observation_space_size)))
            Functions.plot_frozenlake_policy(q_table)
        else:
            Functions.plot_frozenlake_policy(self.strategy.policy)
