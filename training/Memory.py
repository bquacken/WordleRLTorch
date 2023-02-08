# Memory
# Stores results from the networks for training
class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.values = []
        self.dones = []
        self.rewards = []

    def add(self, state, action, value, done, reward):
        self.states.append(state)
        self.actions.append(action)
        self.values.append(value)
        self.dones.append(done)
        self.rewards.append(reward)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.values.clear()
        self.dones.clear()
        self.rewards.clear()

    def _zip(self):
        return zip(self.states,
                   self.actions,
                   self.values,
                   self.dones,
                   self.rewards)

    def __iter__(self):
        for data in self._zip():
            return data

    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data
