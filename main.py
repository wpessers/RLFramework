import gym

from agent import Agent
from environment import Environment
from mdp.MDP import MDP
from strategies.deepqlearning import DeepQLearning
from strategies.doubledeepqlearning import DoubleDeepQLearning
from strategies.montecarlostrategy import MonteCarloStrategy
from strategies.nstepqlearnstrategy import NStepQLearnStrategy
from strategies.qlearnstrategy import QLearnStrategy
from strategies.valueiteration import ValueIteration


def run_frozenlake():
    # region Frozenlake
    env = gym.make("FrozenLake-v0")
    # env = gym.make("CartPole-v0")
    environment = Environment(env)

    qlearnstrat = QLearnStrategy(learning_rate=0.85,
                                 expl_decay_rate=0.01,
                                 discount_factor=0.95,
                                 expl_prob_max=1,
                                 expl_prob_min=0.0001,
                                 environment=environment)

    nstepqlearnstrat = NStepQLearnStrategy(learning_rate=0.8,
                                           expl_decay_rate=0.01,
                                           discount_factor=0.95,
                                           expl_prob_max=1,
                                           expl_prob_min=0.0001,
                                           environment=environment,
                                           steps_amount=10)

    montecarlostrat = MonteCarloStrategy(learning_rate=0.8,
                                         expl_decay_rate=0.01,
                                         discount_factor=0.95,
                                         expl_prob_max=1,
                                         expl_prob_min=0.0001,
                                         environment=environment)

    mdp = MDP(environment.observation_space_size, environment.action_space_size)
    valueiteration = ValueIteration(mdp=mdp,
                                    precision=0.01,
                                    discount_factor=0.95,
                                    expl_prob_max=1,
                                    expl_prob_min=0.0000001,
                                    expl_decay_rate=0.002,
                                    environment=environment,
                                    learning_rate=0.8)

    agent = Agent(environment, qlearnstrat)
    agent.learn(2001)
    # endregion


def run_cartpole():
    # region cartpole (DQN)
    env_cp = gym.make("CartPole-v0")
    environment = Environment(env_cp)

    deepqlearning = DeepQLearning(batch_size=32, update_interval=8, learning_rate=0.001, expl_decay_rate=0.995,
                                  discount_factor=0.95, expl_prob_max=1,
                                  expl_prob_min=0.0000001, environment=environment, max_experience_size=1000)
    doubledeepqlearning = DoubleDeepQLearning(batch_size=32, update_interval=8, learning_rate=0.001,
                                              expl_decay_rate=0.995,
                                              discount_factor=0.95, expl_prob_max=1,
                                              expl_prob_min=0.0000001, environment=environment,
                                              max_experience_size=1000)
    agent = Agent(environment, doubledeepqlearning)
    agent.learn(1001)
    # endregion


def main():
    run_frozenlake()
    # run_cartpole()


if __name__ == "__main__":
    main()
