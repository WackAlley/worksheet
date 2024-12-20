from q_table import QTable
from rps_game.env import RPSEnv
from utilities.player_models import QTableAgent_lookup_only, QTableAgent_with_exploration, MyOponentModel
from utilities.training_and_evaluation import training, evaluation
import time

if __name__ == "__main__":
        start = time.time()
        # example training:
        learning_rate = 0.9
        discount_factor = 0.8
        exploration_prob = 0.1
        eval_epochs = int(1e4)
        train_epochs = int(3e5)
        repetitions = int(10)
        #name = f"training_results/lr{learning_rate}_disc{discount_factor}_expp{exploration_prob}"
        name = "example_qtable"
        qtable = QTable.load(f"{name}.pkl")
        #train against self, player and opponend have same q_table:
        model_self_train = QTableAgent_with_exploration(qtable, exploration_prob)
        model_self_eval = QTableAgent_with_exploration(qtable, 0.001) # not 0 to avoid infinity loops
        model_opponent = QTableAgent_lookup_only(qtable)
        # test against base opponent:
        env_eval = RPSEnv(render_mode=None, opponent=None, size=4, n_holes=2)
        env_train = RPSEnv(render_mode=None, opponent=MyOponentModel(model_opponent), size=4, n_holes=2)
        # opponent = None -> Base Opponent with random actions
        wins, looses = evaluation(env_eval, model_self_eval, eval_epochs)
        print(f"result before: wins: {wins}, looses: {looses}")
        for i in range(repetitions):
                training(env_train,model_self_train, train_epochs)
                # Speichern
                qtable.save(f"{name}_advanced.pkl")
                wins, looses = evaluation(env_eval, model_self_eval, eval_epochs)
                print(f"result after iteration {i}: wins: {wins}, looses: {looses}")
        qtable.save(f"{name}_advanced.pkl")

        env_train.close()
        env_eval.close()
        end = time.time()
        print(end-start)

