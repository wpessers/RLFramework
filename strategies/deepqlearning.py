from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DeepQLearning:
    def __init__(self, batch_size: int, states_amount, actions_amount):
        self.model = self.init_model(states_amount, actions_amount)
        


    def init_model(self, states_amount, actions_amount):
        dqn_model = Sequential()
        dqn_model.add(Dense(24, input_shape=(states_amount,), activation="relu"))
        dqn_model.add(Dense(24, activation="relu"))
        dqn_model.add(Dense(actions_amount, activation="linear"))
        dqn_model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return dqn_model


