import numpy as np
from icecream import ic
from q_table import QTable
from rps_game.env import RPSEnv
from utilities.player_models import QTableAgent_with_exploration
from pandas import DataFrame as df
import sys


learning_rate = 0.8
discount_factor = 0.95
exploration_prob = 0.2
n_epochs = int(1e4)

qtable = QTable(learning_rate=learning_rate, discount_factor=discount_factor, exploration_prob=exploration_prob)
model_self = QTableAgent_with_exploration(qtable, exploration_prob)
env = RPSEnv(render_mode=None, size=4, n_holes=2)
observation, info = env.reset()

wins = 0
looses = 0
for epoch in range(n_epochs):
    observation_this_state, info = env.reset()  # Startzustand
    done = False
    while not done:
        action_mask = env.compute_action_mask()  # Gültige Aktionen
        action, _ = model_self.predict(observation_this_state, action_mask)
        observation_next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if not done:
            action_mask_next_State = env.compute_action_mask()
        else:
            if reward == -1:
                looses += 1
            if reward >= 0.15:
                wins += 1
            action_mask_next_State = None
        model_self.learn(observation_this_state, action, reward, observation_next_state, action_mask_next_State)
        observation_this_state = observation_next_state

    if (epoch <= 2.5e3 and epoch % int(5e2)== 0) or \
            (2.5e3 < epoch <= 1e5 and epoch % int(1e4) == 0) or \
            (epoch > 1e5 and epoch % int(1e5) == 0) :
        exploration_prob = 0 if exploration_prob != 0 else 0.2
        model_self = QTableAgent_with_exploration(qtable, exploration_prob)
        ic("_________________________________________")
        ic("exploration_prob set to: ", exploration_prob)
        ic("evaluation....")
        looses = 0
        wins = 0
    if (epoch <= 2.5e3 and epoch % int(5e2)== int(1e2)) or \
            (2.5e3 < epoch <= 1e5 and epoch % int(1e4) == int(1e3)) or \
            (epoch > 1e5 and epoch % int(1e5) == int(1e4)) :
        total = wins + looses
        ic("evaluation results:", total)
        wins = wins * 100 / total
        looses = looses * 100 / total
        ic(wins, "%")
        ic(looses, "%")
        ic(len(qtable))
        ic(epoch)
        exploration_prob = 0.2
        ic("_________________________________________")
        ic("exploration_prob set to: ", exploration_prob)
        ic("learning.... ")

ic("Größe der Q-Table:", sys.getsizeof(qtable)/1e6, "MB")


qtable_df = df((
    {"state": key[0], "action": key[1], "value": value}
    for key, value in qtable.items()))
qtable_df.to_csv("example_qtable.csv")
#Speichern
qtable.save("example_qtable.pkl")

# Laden und benutzen einer q_table:
example_observation = np.array([[0, 5, 0, 0], [0, -2, 3, 0], [0, 0, 0, -2],[0, 0, 0, 0]])
example_actions = np.array([0, 1, 2, 3, 4])
# Laden
valid_q_values = qtable.get_all_q_given_state(example_observation, example_actions)
print(valid_q_values)
action = example_actions[np.argmax(valid_q_values)]
print(action)
qtable_loaded = QTable.load("example_qtable.pkl")
print(type(qtable_loaded))
env.close()
