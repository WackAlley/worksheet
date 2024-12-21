import numpy as np
from rps_game.env import RPSEnv
from q_table import QTable
import time


class QTableAgentModel():
    def __init__(self, qtable):
        self.qtable = qtable

    def predict(self, observation):
        mask = env.compute_action_mask() #env ist in outher scope definiert, nicht die eleagteste weise ;) suche nach alternative
        valid_actions = np.argwhere(mask).flatten()
        valid_q_values = self.qtable.get_all_q_given_state(observation, valid_actions)
        return valid_actions[np.argmax(valid_q_values)], None


if __name__ == "__main__":

    env = RPSEnv(render_mode=None,size=4, n_holes=2)
    #render_mode: "human" | "console" | None

    observation, info = env.reset(seed=42)

    start = time.time()
    #n_epochs = int(3e6)
    n_epochs = int(1e5)
    print(n_epochs)


    #learning_rate = 0.9
    #discount_factor = 0.7
    #exploration_prob = 0.2

    env = RPSEnv(render_mode=None, size=4, n_holes=2)
    # render_mode: "human" | "console" | None

    observation, info = env.reset()

    print("learning_rate, discount_factor, exploration_prob, wins, looses")
    for learning_rate in [0.8,0.9]:
        for discount_factor in [0.7,0.8,0.9,0.95]: #
            for exploration_prob in [0.1,0.15,0.2,0.3]:
                name = f"training_results/lr{learning_rate}_disc{discount_factor}_expp{exploration_prob}"
                qtable = QTable.load(f"{name}.pkl")
                for i in range(3):
                    wins = 0
                    looses = 0
                    for epoch in range(int(n_epochs)):
                        observation_this_state, info = env.reset()  # Startzustand
                        done = False
                        while not done:
                            mask = env.compute_action_mask()  # Gültige Aktionen
                            valid_actions = np.argwhere(mask).flatten()  # Indizes gültiger Aktionen, entspricht Aktion
                            valid_q_values = qtable.get_all_q_given_state(observation_this_state, valid_actions)
                            action = valid_actions[np.argmax(valid_q_values)]

                            # Wechsel zum nächsten Zustand
                            observation_next_state, reward, terminated, truncated, info = env.step(int(action))
                            done = terminated or truncated
                            if not done:
                                mask = env.compute_action_mask()
                                valid_actions_next_state = np.argwhere(mask).flatten()
                            else:
                                if reward == -1:
                                    looses += 1
                                if reward > 0.15:
                                    wins += 1
                                valid_actions_next_state = ("f")
                            qtable.update_table(observation_this_state, action, observation_next_state,
                                               valid_actions_next_state, reward)
                            # ic("updated")
                            observation_this_state = observation_next_state

                    print(f"{learning_rate}, {discount_factor}, {exploration_prob}, {wins}, {looses}")
        #end = time.time()
        #print(end-start)
    env.close()




