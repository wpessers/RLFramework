import gym
import numpy as np

from agent import Agent
from environment import Environment
from strategies.deepqlearning import DeepQLearning
from strategies.montecarlostrategy import MonteCarloStrategy
from strategies.nstepqlearnstrategy import NStepQLearnStrategy
from strategies.qlearnstrategy import QLearnStrategy


def main():
    env = gym.make("FrozenLake-v0")
    environment = Environment(env)
    qlearnstrat = QLearnStrategy(learning_rate=0.8, expl_decay_rate=0.01, discount_factor=0.95, expl_prob_max=1,
                                 expl_prob_min=0.0001, environment=environment)
    nstepqlearnstrat = NStepQLearnStrategy(learning_rate=0.8, expl_decay_rate=0.01, discount_factor=0.95,
                                           expl_prob_max=1, expl_prob_min=0.0001, environment=environment,
                                           steps_amount=10)
    montecarlostrat = MonteCarloStrategy(learning_rate=0.8, expl_decay_rate=0.01, discount_factor=0.95, expl_prob_max=1,
                                         expl_prob_min=0.0001, environment=environment)

    deepqlearning = DeepQLearning(batch_size=5, update_interval= 10,learning_rate=0.8, expl_decay_rate=0.01, discount_factor=0.95, expl_prob_max=1,
                                 expl_prob_min=0.0001, environment=environment)
    agent = Agent(environment, deepqlearning)
    agent.learn(2001)


if __name__ == "__main__":
    main()
