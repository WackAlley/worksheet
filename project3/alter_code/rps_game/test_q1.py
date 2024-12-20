import numpy as np
import random
import pandas as pd
from icecream import ic

#from rps_game.test_q import get_nextstate

ic("test")
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
        # Aktion auswählen (ε-greedy)
        
        max_q = np.max(self.get_all_q_given_state(next_state, possible_actions_in_next_state)) 
        q_diff = self.discount_factor * max_q - self[(state, action)]
        self[(state, action)] += self.learning_rate * (reward + q_diff)


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

class Envirement():
    def __init__(self, hole_config):
        self.current_state, self.my_pos = get_random_state(hole_config=hole_config)
        #self.terminated = False

    def next_turn(self):
        #result = self.current_state.copy()
        #result[self.current_state >= 4] -= 3
        #result[(self.current_state >= 1) & (self.current_state <= 3)] += 3
        #self.current_state = result
        idx_y, idx_x = np.unravel_index(np.argmax(self.current_state), self.current_state.shape)
        self.current_state = np.select(
            [(self.current_state >= 1) & (self.current_state <= 3), self.current_state >= 4],
            [self.current_state + 3, self.current_state - 3],
           default=self.current_state)
        self.my_pos = Position(idx_x, idx_y)

    @staticmethod
    def get_suit(value):
        return value % 3

    @staticmethod
    def get_next_pos(my_pos, action):
        dict_actions = {0: Position(1, 0),  # move right
                        1: Position(-1, 0),  # move left
                        2: Position(0, 1),  # move down
                        3: Position(0, -1),  # move up
                        4: Position(0, 0)  # convert
                        }
        move = dict_actions[action]
        return my_pos + move

    def get_next_state(self, action):
        next_pos = self.get_next_pos(self.my_pos, action)
        next_state = self.current_state.copy()
        reward = self.get_reward(action)
        if action != 4: #move
            if reward >= -0.05:  # did not lose, update position of my agent
                next_state[next_pos.y, next_pos.x] = self.current_state[self.my_pos.y, self.my_pos.x]
            next_state[self.my_pos.y, self.my_pos.x] = 0 # peace moved, reset old entry
        else: #convert
            next_state[self.my_pos.y, self.my_pos.x] = (next_state[self.my_pos.y, self.my_pos.x] % 3) + 1  #1 -> 1+1 =2 ,2 -> 2+1 =3 ,3 -> 0+1 =1
        return next_state

    def move(self, action):
        #if self.get_reward(self.current_state, action, self.my_pos)  <= -0.9: #lose
            #self.terminated = True
        self.current_state = self.get_next_state(action)
        self.my_pos = self.get_next_pos(self.my_pos, action)


    def get_reward(self, action):
        reward = -0.01 # time penalty
        #my_pos1, my_pos2 = my_pos
        next_pos = self.get_next_pos(self.my_pos, action)
        if self.current_state[next_pos.y, next_pos.x] == -2: #lose
            reward = -1
        if 4 <= self.current_state[next_pos.y, next_pos.x] <= 6:  # collision
            reward = (self.get_suit(self.current_state[self.my_pos.y, self.my_pos.x]) - self.get_suit(self.current_state[next_pos.y, next_pos.x]) + 4) % 3 - 1
        #ic("reward_state", state)
        #ic("reward_state", action)
        #ic("reward_my_pos", my_pos)
        #ic("reward", reward)
        return reward


    def get_possible_actions(self,my_pos):
        #ypos, xpos = my_pos
        actions = []
        neighbourhood = []
        if my_pos.x != 3:
            next_pos = my_pos + Position(1, 0)
            #same suit oponent on next pos, invalid move:
            if self.current_state[next_pos.y, next_pos.x] != self.current_state[my_pos.y, my_pos.x] +3:
                actions.append(0)
            neighbourhood.append(next_pos)
        if my_pos.x != 0:
            next_pos = my_pos + Position(-1, 0)
            if self.current_state[next_pos.y, next_pos.x] != self.current_state[my_pos.y, my_pos.x] + 3:
                actions.append(1)
            neighbourhood.append(next_pos)
        if my_pos.y != 3:
            next_pos = my_pos + Position(0, 1)
            if self.current_state[next_pos.y, next_pos.x] != self.current_state[my_pos.y, my_pos.x] + 3:
                actions.append(2)
            neighbourhood.append(next_pos)
        if my_pos.y != 0:
            next_pos = my_pos + Position(0, -1)
            if self.current_state[next_pos.y, next_pos.x] != self.current_state[my_pos.y, my_pos.x] + 3:
                actions.append(3)
            neighbourhood.append(next_pos)
        #self.current_state
        ##two pieces are of the same type, they cannot move to the same field.
        #if(self.current_state[pos.y, pos.x] >= 0 ):

        if all(self.current_state[pos.y, pos.x] <= 0 for pos in neighbourhood): #if no other peace is in neighbourhood
            actions.append(4)
        return actions

    def reset(self, hole_config):
        self.current_state, self.my_pos = get_random_state(hole_config=hole_config)
        #self.terminated =False


"""
for i in range(1000000000):
    s = get_random_state(hole_config=((1, 1), (2, 3)))
    if np.max(s[0]) <= 3:
        print("error")
        print(s)
        print(np.max(s[0]))
exit()
"""


ic("yf")
learning_rate = 0.8
discount_factor = 0.95
exploration_prob = 0.2
#nEpochs = int(2e7)
nEpochs = int(3)
hole_config=((1, 1), (2, 3))
ic()

qtable = Q_table(learning_rate = 0.8, discount_factor = 0.95, exploration_prob = 0.2)
env = Envirement(hole_config)
ic("initial state", env.current_state)


for i in range(nEpochs):
    #current_state, my_pos = get_random_state(hole_config=hole_config)
    reward = 0
    while -0.05 <= reward <= 0.05:
        possible_actions = env.get_possible_actions(env.my_pos)
        if np.random.rand() < exploration_prob:
            action = random.choice(possible_actions)
            ic("picked randomly: ", action)
        else:
            action = possible_actions[np.argmax(qtable.get_all_q_given_state(env.current_state, possible_actions))]
        ic(possible_actions)
        ic(action)
        reward = env.get_reward(action)
        ic(reward)
        next_state = env.get_next_state(action)
        next_pos = env.get_next_pos(env.my_pos, action)
        possible_actions_in_next_state = env.get_possible_actions(next_pos)
        qtable.update_table(env.current_state, action,next_state, possible_actions_in_next_state, reward)
        ic("before move:",env.current_state, env.my_pos)
        env.move(action)
        ic("after move:", env.current_state, env.my_pos)
        env.next_turn()
        ic("after next turn:", env.current_state, env.my_pos)
    env.reset(hole_config)
    ic("new random state:", env.current_state)

    if i % 1e3 == 0:
        print(i, "of", nEpochs )
        print(len(qtable))


    #print(random_state, current_state[my_pos1, my_pos2], my_xpos, my_ypos, get_possible_moves(my_pos))

    #qtable_df = pd.DataFrame.from_dict(qtable.to_dict)
qtable_df = pd.DataFrame((
{"state": key[0], "action": key[1], "value": value}
for key, value in qtable.items()))
qtable_df.to_csv("qtable.csv")
# (14×13×3×3) × ((3÷14)×5+(7÷14)×4+(4÷14)×3) = 6435 = Total number of states, with hole_config: ((1, 1), (2, 3))





