import numpy as np
import random
import pandas as pd
from icecream import ic

#from rps_game.gameq import reward
#from collections import defaultdict

class Q_table(dict):
    def __init__(self, learning_rate, discount_factor, exploration_prob):
        super().__init__()#(lambda: 0)
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
        if key in self:
            return super().__getitem__(key)
        else:
            return True
        #key = (self._make_numpy_array_hashable(state), action)
        #return super().__getitem__(key)

    def update_table(self, state, action, next_state, possible_actions_in_next_state, reward):
        max_q = np.max(self.get_all_q_given_state(next_state, possible_actions_in_next_state)) #??
        q_diff = self.discount_factor * max_q - self[(state, action)]
        self[(state, action)] = self.learning_rate * (reward + q_diff)


    def get_all_q_given_state(self, state, possible_actions):
        #return np.asarray([self.__getitem__((state.copy(), 0))])
        return np.asarray([self[(state , action)] for action in possible_actions])

    @staticmethod
    def _make_numpy_array_hashable(np_array):
        return tuple(np_array.flatten())

class Position:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Position(x={self.x}, y={self.y})"

    # Operator überladen: Addition
    def __add__(self, other):
            return Position(self.x + other.x, self.y + other.y)

    # Operator überladen: Subtraktion
    def __sub__(self, other):
            return Position(self.x - other.x, self.y - other.y)


def get_random_state(hole_config = None):
    random_state = np.zeros((4, 4), dtype=int)
    if hole_config is None: #two holes at random pos
        values = [-2, -2, np.random.randint(1,4), np.random.randint(4,7)]
    else:
        values = [np.random.randint(1, 4), np.random.randint(4, 7)]
        random_state[hole_config[0][0], hole_config[0][1]] = -2
        random_state[hole_config[1][0], hole_config[1][1]] = -2
    #values = [-2, -2, np.random.randint(1, 4), np.random.randint(4, 7)]
    for idx, value in enumerate(values):
        while True:
            xpos = np.random.randint(0, 4)
            ypos = np.random.randint(0, 4)
            if random_state[ypos, xpos] == 0:
                break
        random_state[ypos, xpos] = value
        if (idx == 2 and hole_config is None) or (idx == 0 and hole_config is not None) :
            my_pos = Position(xpos,ypos)
    return random_state, my_pos


def get_random_action():
    return np.random.randint(0, 5)


def get_suit(value):
    return value % 3


def get_next_pos(my_pos, action):
    dict_actions = {0: Position(1, 0),  # move right
                    1: Position(-1, 0),  # move left
                    2: Position(0, 1),  # move down
                    3: Position(0, -1),  # move up
                    4: Position(0, 0)  # convert
                    }
    move = dict_actions[action]
    return my_pos + move

def get_nextstate(current_state, current_pos, action):
    next_pos = get_next_pos(current_pos, action)
    next_state = current_state.copy()
    reward = get_reward(current_state, action, current_pos)
    if -0.05 <= reward <= 0.05:
        next_state[next_pos.y, next_pos.x] = 0
    if reward >= 0.9:#won
        next_state[next_pos.y, next_pos.x] = current_state[next_pos.y, next_pos.x]
    next_state[next_pos.y, next_pos.x] = 0
    return next_state


def get_reward(state, action, my_pos):
    reward = -0.01 # time penalty
    #my_pos1, my_pos2 = my_pos
    next_pos = get_next_pos(my_pos, action)
    if state[next_pos.y, next_pos.x] == -2: #lose
        reward = -1
    if 4 <= state[next_pos.y, next_pos.x] <= 6:  # collision
        reward = (get_suit(state[my_pos.y, my_pos.x]) - get_suit(state[next_pos.y, next_pos.x]) + 4) % 3 - 1
    return reward


def get_possible_actions(my_pos):
    #ypos, xpos = my_pos
    actions = []
    if my_pos.x != 3:
        actions.append(0)
    if my_pos.x != 0:
        actions.append(1)
    if my_pos.y != 3:
        actions.append(2)
    if my_pos.y != 0:
        actions.append(3)
    actions.append(4)
    return actions

"""
for i in range(1000000000):
    s = get_random_state(hole_config=((1, 1), (2, 3)))
    if np.max(s[0]) <= 3:
        print("error")
        print(s)
        print(np.max(s[0]))
exit()
"""



learning_rate = 0.8
discount_factor = 0.95
exploration_prob = 0.2
nEpochs = int(2e7)
hole_config=((1, 1), (2, 3))

qtable = Q_table(learning_rate = 0.8, discount_factor = 0.95, exploration_prob = 0.2)

for i in range(nEpochs):
    current_state, my_pos = get_random_state(hole_config=hole_config)
    possible_actions = get_possible_actions(my_pos)
    reward = 0
    #while reward == 0:
    if np.random.rand() < exploration_prob:
        action = random.choice(possible_actions)
    else:
        action = possible_actions[np.argmax(qtable.get_all_q_given_state(current_state, possible_actions))]
    reward = get_reward(current_state, action, my_pos)
    next_state = get_nextstate(current_state, my_pos, action)
    next_pos = get_next_pos(my_pos, action)
    possible_actions_in_next_state = get_possible_actions(next_pos)
    qtable.update_table(current_state, action,next_state, possible_actions_in_next_state, reward)
    if i % 1e6 == 0:
        print(i, "of", nEpochs )
        print(len(qtable))

    #print(random_state, current_state[my_pos1, my_pos2], my_xpos, my_ypos, get_possible_moves(my_pos))

    #qtable_df = pd.DataFrame.from_dict(qtable.to_dict)
qtable_df = pd.DataFrame((
{"state": key[0], "action": key[1], "value": value}
for key, value in qtable.items()))
qtable_df.to_csv("qtable.csv")
# (14×13×3×3) × ((3÷14)×5+(7÷14)×4+(4÷14)×3) = 6435 = Total number of states, with hole_config: ((1, 1), (2, 3))





