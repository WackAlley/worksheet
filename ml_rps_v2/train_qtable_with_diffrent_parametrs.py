import numpy as np
from rps_game.env import RPSEnv
from q_table import QTable
import time
from pandas import DataFrame as df

from utilities.training_and_evaluation import training, evaluation
from utilities.player_models import QTableAgent_lookup_only, QTableAgent_with_exploration, MyOponentModel

if __name__ == "__main__":
    evaluation_epochs = int(1e5)
    training_epochs = int(2e6)

    start = time.time()

    env = RPSEnv(render_mode=None,size=4, n_holes=2)
    #render_mode: "human" | "console" | None

    print("learning_rate, discount_factor, exploration_prob, wins, looses")
    for learning_rate in [0.8,0.9]:
        for discount_factor in [0.7,0.8,0.9,0.95]: #
            for exploration_prob in [0.1,0.15,0.2,0.3]:
                qtable = QTable(learning_rate=learning_rate, discount_factor=discount_factor,
                                exploration_prob=exploration_prob)
                agent_model = QTableAgent_with_exploration(qtable, exploration_prob)
                #training
                training(env=env, agent_model= agent_model, n_epochs= training_epochs)
                #save results:
                qtable_df = df((
                    {"state": key[0], "action": key[1], "value": value}
                    for key, value in qtable.items()))
                name = f"lr{learning_rate}_disc{discount_factor}_expp{exploration_prob}"
                qtable_df.to_csv(f"{name}.csv")
                qtable.save(f"{name}.pkl")
                #evaluation
                wins, looses = evaluation(env=env,agent_model=agent_model, n_epochs=evaluation_epochs)
                print(f"{learning_rate}, {discount_factor}, {exploration_prob}, {wins}, {looses}")
    env.close()
    end = time.time()
    print(end-start)




