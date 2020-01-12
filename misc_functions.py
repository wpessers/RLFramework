import numpy as np
import matplotlib.pyplot as plt

from lrl import environments
from lrl.utils import plotting


class Functions:
    @staticmethod
    def argmax_float(a):
        return np.random.choice(np.flatnonzero(np.isclose(a, a.max())))

    @staticmethod
    def plot_frozenlake_policy(policy):
        frozenlake = environments.frozen_lake.RewardingFrozenLakeEnv(map_name='4x4', is_slippery=True)
        preferred_actions_dict = dict(enumerate(np.argmax(policy, axis=1)))
        plotting.plot_solver_results(env=frozenlake, policy=preferred_actions_dict, color='red')
        plt.show()

    @staticmethod
    def plot_success_rate(x_vals, y_vals):
        plt.plot(x_vals, y_vals)
        plt.xlabel("episode")
        plt.ylabel("succes rate")
        plt.show()
