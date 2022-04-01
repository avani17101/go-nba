
from gym.spaces import Discrete
import numpy as np
# source: https://stackoverflow.com/questions/45001361/is-there-a-way-to-implement-an-openais-environment-where-the-action-space-chan

class IterableDiscrete(Discrete):
    """
    wrapper for variable action space
    """
    def __init__(self, n):
        super().__init__(n)
        self.index = 0
        #initially all actions are available
        self.available_actions = np.arange(0, n)

    def disable_actions(self, actions):
        """ You would call this method inside your environment to remove available actions"""
        self.available_actions = [action for action in self.available_actions if action not in actions]
#         print("after disabling",self.available_actions )
        return self.available_actions

    def enable_actions(self, actions):
        """ You would call this method inside your environment to enable actions"""
        self.available_actions = np.append(self.available_actions,actions)
#         print("after enabling",self.available_actions )
        return self.available_actions

    def sample(self):
#         print("sampling",np.random.choice(self.available_actions))
        return np.random.choice(self.available_actions)

    def contains(self, x):
        return x in self.available_actions


    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < self.n:
            x = self.index
            self.index += 1
            return x
        else:
            raise StopIteration

    def __repr__(self):
        return "IterableDiscrete(%d)" % self.n

    def __eq__(self, other):
        return isinstance(other, IterableDiscrete) and self.n == other.n
