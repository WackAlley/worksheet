import numpy as np
import time
#from rps_game.env import RPSEnv
from q_table import QTable
from icecream import ic
from pandas import DataFrame as df
import sys
import time
from rps_v2.rps.rps_game.opponent import ModelOpponent
from rps_v2.rps.rps_game.env import RPSEnv
from rps_v2.rps.rps_game.env import Game, Obstacle


class QTableAgentModel():
    def __init__(self, qtable):
        self.qtable = qtable

    def predict(self, observation):
        mask = env.compute_action_mask() #env ist in outher scope definiert, nicht die eleagteste weise ;) suche nach alternative
        valid_actions = np.argwhere(mask).flatten()
        valid_q_values = self.qtable.get_all_q_given_state(observation, valid_actions)
        return valid_actions[np.argmax(valid_q_values)], None


if __name__ == "__main__":
    path_op_q_table = "qtable.pkl"
    qtable_opponent = QTable.load(path_op_q_table)
    opponent_model = QTableAgentModel(qtable_opponent)
    opponent = ModelOpponent(opponent_model)
    #opponent.maskable = True

    env = RPSEnv(render_mode="human", opponent=opponent,size=4, n_holes=2)
    #render_mode: "human" | "console" | None

    observation, info = env.reset(seed=42)

    learning_rate = 0.8 #0.1
    discount_factor = 0.95 #0.99
    exploration_prob = 0.2 #0.1
    qtable = QTable(learning_rate=learning_rate, discount_factor=discount_factor, exploration_prob=exploration_prob)

    #n_epochs = int(2e6)
    n_epochs = int(2e4)
    wins = 0
    looses = 0
    for epoch in range(n_epochs):

        observation_this_state, info = env.reset()  # Startzustand
        done = False
        while not done:
            mask = env.compute_action_mask() # Gültige Aktionen
            valid_actions = np.argwhere(mask).flatten() # Indizes gültiger Aktionen, entspricht Aktion

            if np.random.rand() < exploration_prob:
                action = np.random.choice(valid_actions)
                #ic(action)
                #ic("picked randomly: ", action) # Zufällige Aktion
                #ic(type(action))
            else:
                valid_q_values = qtable.get_all_q_given_state(observation_this_state, valid_actions)
                action = valid_actions[np.argmax(valid_q_values)]
                #ic(action)
                #ic(type(action))

            # Wechsel zum nächsten Zustand
            observation_next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            if not done:
                mask = env.compute_action_mask()
                valid_actions_next_state = np.argwhere(mask).flatten()
            else:
                if reward == -1:
                    looses += 1
                if reward == +1:
                    wins += 1
                valid_actions_next_state = ("f")
            qtable.update_table(observation_this_state, action, observation_next_state, valid_actions_next_state, reward)
            #ic("updated")

            observation_this_state = observation_next_state

        if (epoch <= 2.5e3 and epoch % int(5e2)== 0) or \
                (2.5e3 < epoch <= 1e5 and epoch % int(1e4) == 0) or \
                (epoch > 1e5 and epoch % int(1e5) == 0) :
            exploration_prob = 0 if exploration_prob != 0 else 0.2
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
    qtable_df.to_csv("qtable.csv")

    #Speichern
    qtable.save("qtable.pkl")

    example_observation = np.array([[0, 5, 0, 0], [0, -2, 3, 0], [0, 0, 0, -2],[0, 0, 0, 0]])
    example_actions = np.array([0, 1, 2, 3, 4])
    # Laden
    valid_q_values = qtable.get_all_q_given_state(example_observation, example_actions)
    print(valid_q_values)
    action = example_actions[np.argmax(valid_q_values)]
    print(action)
    qtable_loaded = QTable.load("qtable.pkl")
    print(type(qtable_loaded))
    env.close()




