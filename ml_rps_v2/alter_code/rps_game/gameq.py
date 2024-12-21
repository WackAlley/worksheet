import numpy as np
import random
import pandas as pd


class Q_table():
    def __init__(self):
        self.table = {} # dictionary

    def update(self, key, newvalue):
        state, move = key
        key = (self._make_numpy_array_hashable(state), move)
        self.table[key] = newvalue
    
    def getvalue(self, key):
        state, move = key
        key = (self._make_numpy_array_hashable(state), move)
        if tuple(key) in self.table:
            return self.table[tuple(key)]
        else:
            return 0
    def getallq_givenstate(self, state, possiblemoves):
        state = self._make_numpy_array_hashable(state)
        return np.asarray((self.table[(state, move)] for move in possiblemoves))

    @staticmethod
    def _make_numpy_array_hashable(np_array):
        return tuple(np_array.flatten())


#if hole_config = ((1, 1), (2, 3)):
def get_random_state(hole_config = None):
    #my_xpos
    #my_ypos
    random_state = np.zeros((4,4), dtype=int)
    if hole_config != None:
        values = [-2, -2, np.random.randint(1,4), np.random.randint(4,7)]
    else:
        values = [np.random.randint(1, 4), np.random.randint(4, 7)]
        random_state[hole_config[0][0], hole_config[0][1]] = -2
        random_state[hole_config[1][0], hole_config[1][1]] = -2
    for idx, value in enumerate(values):
        xpos = np.random.randint(0,4)
        ypos = np.random.randint(0,4)
        while random_state[xpos, ypos] != 0:
            xpos = np.random.randint(0,4)
            ypos = np.random.randint(0,4)
        random_state[xpos, ypos] = value
        if idx == 2:
            my_xpos = xpos
            my_ypos = ypos
    return random_state, my_ypos, my_xpos

def get_random_action():
    return np.random.randint(0,5)

def get_suit(value):
    return value % 3

def get_nextpos(my_pos, action):
    dict_actions = {0 : (0,1),  # move right
                    1 : (0,-1),  # move left
                    2 : (1,0), # move down
                    3 : (-1,0), # move up
                    4 : (0,0) # convert
                    }
    print(my_pos)
    print(dict_actions[action])
    action1, action2 = dict_actions[action]
    mypos1, mypos2 = my_pos
    print(my_pos)
    return (mypos1 + action1, mypos2 + action2)
   # return (sum(x) for x in zip(my_pos,dict_actions[action]))

def get_nextstate(current_state, current_pos, action):
    next_pos1, next_pos2 = get_nextpos(current_pos, action)
    next_state = current_state
    current_pos1, current_pos2 = current_pos
    if get_reward(current_state, action, current_pos) == 0:
        next_state[next_pos1, next_pos2] = 0
    if get_reward(current_state, action, current_pos) == 1:
        next_state[next_pos1, next_pos2] = next_state[current_pos1, current_pos2]
    next_state[current_pos1, current_pos2] = 0
    return next_state

def get_reward(state, action, my_pos): 
    reward = 0
    my_pos1, my_pos2 = my_pos
    nextpos = get_nextpos(my_pos, action)
    #print(a,b:= *nextpos, "\n")
    #print(state[0,0])
    pos1, pos2 = nextpos
    print(pos1, pos2)
    print(state[pos1,pos2])
    print(state[pos1,pos2]>=4)
    print(state[pos1,pos2] == -2)
    if state[pos1, pos2] == -2:
        reward = -1
    if state[pos1, pos2]>= 4 and state[pos1, pos2] <= 6: # collision
        reward = (get_suit(state[my_pos1, my_pos2]) - get_suit(state[pos1, pos2]) + 4) % 3 - 1

    return reward
        
def get_possible_moves(my_pos):
    ypos, xpos = my_pos
    actions = []
    if xpos != 3:
        actions.append(0)
    if xpos != 0:
        actions.append(1)
    if ypos != 3:
        actions.append(2)
    if ypos != 0:
        actions.append(3)
    actions.append(4)
    return actions

learning_rate    = 0.8
discount_factor  = 0.95
exploration_prob = 0.2
nEpochs = 10

qtable = Q_table()

for i in range(10):
    s = get_random_state(hole_config=((1, 1), (2, 3)))
    print(s)
#exit()


for i in range(nEpochs):
    current_state, my_xpos, my_ypos = get_random_state()
    my_pos1, my_pos2 = my_pos = (my_ypos, my_xpos)
    possiblemoves = get_possible_moves(my_pos)
    if np.random.rand() < exploration_prob: 
        move = random.choice(possiblemoves)
    else: 
        move = np.argmax(qtable.getallq_givenstate(current_state, possiblemoves))
    reward = get_reward(current_state, move, my_pos)
    q_value =  discount_factor*np.max(qtable.getallq_givenstate(get_nextstate(current_state, my_pos, move), possiblemoves )
    newqvalue = qtable.getvalue((current_state, move)) + learning_rate * (reward + q_value) - qtable.getvalue((current_state, move)) ))
    qtable.update((current_state, move), newqvalue)
    print(random_state, current_state[my_pos1, my_pos2], my_xpos, my_ypos, get_possible_moves(my_pos))


qtable_df = pd.DataFrame.from_dict(qtable)
qtable_df.to_csv("qtable.csv")


       



