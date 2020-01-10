import gym
import numpy as np
import matplotlib.pyplot as plt

from agent import Agent
from environment import Environment
from mdp.MDP import MDP
from misc_functions import Functions
from strategies.deepqlearning import DeepQLearning
from strategies.montecarlostrategy import MonteCarloStrategy
from strategies.nstepqlearnstrategy import NStepQLearnStrategy
from strategies.qlearnstrategy import QLearnStrategy
from strategies.valueiteration import ValueIteration


def main():
    '''
    #region frozenlake
    env_fl = gym.make("FrozenLake-v0")
    frozen_lake = Environment(env_fl)
    qlearnstrat = QLearnStrategy(learning_rate=0.85, expl_decay_rate=0.01, discount_factor=0.95, expl_prob_max=1,
                                 expl_prob_min=0.0001, environment=frozen_lake)
    nstepqlearnstrat = NStepQLearnStrategy(learning_rate=0.8, expl_decay_rate=0.01, discount_factor=0.95,
                                           expl_prob_max=1, expl_prob_min=0.0001, environment=frozen_lake,
                                           steps_amount=10)
    montecarlostrat = MonteCarloStrategy(learning_rate=0.8, expl_decay_rate=0.01, discount_factor=0.95, expl_prob_max=1,
                                         expl_prob_min=0.0001, environment=frozen_lake)

    deepqlearning = DeepQLearning(batch_size=5, update_interval= 10,learning_rate=0.8, expl_decay_rate=0.01, discount_factor=0.95, expl_prob_max=1,
                                 expl_prob_min=0.0001, environment=frozen_lake)
    mdp = MDP(frozen_lake.observation_space.n, frozen_lake.action_space.n)
    valueiteration = ValueIteration(mdp=mdp, precision=0.01, expl_decay_rate=0.01, discount_factor=0.95, expl_prob_max=1, expl_prob_min=0.0001)
    agent = Agent(frozen_lake, deepqlearning)
    agent.learn(2001)
    Functions.plot_frozenlake_policy()
    #endregion
    '''

    #region cartpole (DQN)
    env_cp = gym.make("CartPole-v1")
    cart_pole = Environment(env_cp)
    deepqlearning = DeepQLearning(batch_size=5, update_interval=10, learning_rate=0.8, expl_decay_rate=0.01,
                                  discount_factor=0.95, expl_prob_max=1,
                                  expl_prob_min=0.0001, environment=cart_pole)
    agent = Agent(cart_pole, deepqlearning)
    agent.learn(2001)
    #endregion
if __name__ == "__main__":
    main()
