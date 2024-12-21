import numpy as np
from icecream import ic
import pickle
class QTable(dict):
    def __init__(self, learning_rate, discount_factor, exploration_prob):
        super().__init__()  # (lambda: 0)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob

    def __setitem__(self, key, value):
        state, action = key
        key = (self._make_numpy_array_hashable(state), action)
        super().__setitem__(key, value)

    def __getitem__(self, key):
        state, action = key
        key = (self._make_numpy_array_hashable(state), action)
        #ic(key, type(key),type(key[0]), type(key[1]))
        if key in self:
            return super().__getitem__(key)
        else:
            return True
        # key = (self._make_numpy_array_hashable(state), action)
        # return super().__getitem__(key)

    def update_table(self, state, action, next_state, possible_actions_in_next_state, reward):
        # Aktion auswählen (ε-greedy)
        #max_q = np.max(self.get_all_q_given_state(next_state, possible_actions_in_next_state))
        #q_diff = self.discount_factor * max_q - self[(state, action)]
        #self[(state, action)] += self.learning_rate * (reward + q_diff)
        max_q = np.max(self.get_all_q_given_state(next_state, possible_actions_in_next_state))
        target = reward + self.discount_factor * max_q
        self[(state, action)] += self.learning_rate * (target - self[(state, action)])


    def get_all_q_given_state(self, state, possible_actions):
        # return np.asarray([self.__getitem__((state.copy(), 0))])
        #ic()
        #ic(possible_actions)
        #ic(type(possible_actions))
        return np.asarray([self[(state, action)] for action in possible_actions])

    @staticmethod
    def _make_numpy_array_hashable(np_array):
        if isinstance(np_array, np.ndarray):
            return tuple(np_array.flatten())
        if isinstance(np_array, tuple):
            return np_array

    def save(self, filepath):
        """Speichert die Q-Table"""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        """
        #save super class (normal dict):
        with open(filepath + "_dict", "wb") as f:
            pickle.dump(dict(self),f)
        """

    @classmethod
    def load(cls, filepath):
        """Lädt eine Qtable aus einer pkl Datei."""
        with open(filepath, "rb") as f:
            q_table = pickle.load(f)
            "if you encounter problems while loading, double check: Is the class definition (this file) in the same sub\
             directory as it was, while saving (dumping)? If not move class definition."
        # this requires, that the q_table was loaded successfully:
        if not isinstance(q_table, cls):
            raise TypeError(f"Loaded object is not of type {cls.__name__}")
        return q_table
