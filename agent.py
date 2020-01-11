import numpy as np
from gym.envs.classic_control import CartPoleEnv
from gym.wrappers import TimeLimit
from keras.utils import to_categorical

from environment import Environment
from misc_functions import Functions
from percept import Percept
from strategies.learningstrategy import LearningStrategy


class Agent:
    def __init__(self, environment: Environment, strategy: LearningStrategy):
        self.environment = environment
        self.strategy = strategy

    def learn(self, n_episodes: int):
        episode_count = 0

        #stats variables:
        finished_count_100eps = 0
        x_vals = []
        y_vals = []
        scores = []

        if isinstance(self.environment.env, TimeLimit):
            while episode_count < n_episodes:
                current_state = self.environment.reset()
                self.environment.render()
                score_count = 0
                percept = Percept(0, 0, 0, 0, False)
                while not percept.done:
                    self.environment.render()
                    action = self.strategy.next_action(current_state)
                    next_state, reward, done = self.environment.step(action)
                    percept = Percept(current_state, action, next_state, reward, done)
                    self.strategy.learn(percept, episode_count)
                    current_state = percept.next_state
                    score_count += percept.reward

                # print("Agent died, got score of: " + str(score_count))
                scores.append(score_count)
                print(score_count)
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
                    print("\n\nEpisode: " + str(episode_count))
                    print("Exploration probability: " + str(self.strategy.expl_prob))
                    print("Policy:\n")
                    print(self.strategy.policy)
                    print("Amount of times the goal was reached last 100 episodes:\n" + str(finished_count_100eps))
                    x_vals.append(episode_count)
                    y_vals.append(finished_count_100eps)
                    finished_count_100eps = 0

                episode_count += 1

            Functions.plot_success_rate(x_vals, y_vals)


    # def learn(self, n_episodes: int):
    #     episode_count = 0
    #     scores = []
    #     while episode_count < n_episodes:
    #         current_state = self.environment.reset()
    #         # self.environment.render()
    #         score_count = 0
    #         percept = Percept(0, 0, 0, 0, False)
    #         while not percept.done:
    #             # self.environment.render()
    #             action = self.strategy.next_action(current_state)
    #             next_state, reward, done = self.environment.step(action)
    #             percept = Percept(current_state, action, next_state, reward, done)
    #             self.strategy.learn(percept, episode_count)
    #             current_state = percept.next_state
    #             score_count += percept.reward
    #
    #         # print("Agent died, got score of: " + str(score_count))
    #         scores.append(score_count)
    #         print(score_count)
    #         if episode_count > 0 and episode_count % 10 == 0:
    #
    #             print("\nEpisode: " + str(episode_count))
    #             print("Exploration probability: " + str(self.strategy.expl_prob))
    #             print("Average score:" + str(np.mean(scores)))
    #             scores.clear()
    #
    #         episode_count += 1
