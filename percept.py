class Percept:
    def __init__(self, current_state: int, action: int, next_state: int, reward: float, done: bool):
        self.current_state = current_state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.done = done

    def __repr__(self):
        return "Percept:\n\tState: %d\n\tAction: %d\n\tNext State: %d\n\tReward: %f\n\tDone: %r" % (
        self.current_state, self.action, self.next_state, self.reward, self.done)
