from rps_game.env import RPSEnv
import time
import gymnasium as gym
import numpy as np
from rps_game.opponent import ModelOpponent, RandomOpponent
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from utilities.training_and_evaluation import evaluation
from rps_game.opponent import BaseOpponent
from icecream import ic
from joblib import Parallel, delayed
from joblib import Parallel as parallel
from torch.nn import ReLU


def mask_fn(env: gym.Env) -> np.ndarray:
    """
    Generiert eine Aktionsmaske basierend auf der Umgebung.
    """
    return env.compute_action_mask()

policy_kwargs =  dict(
    net_arch=[dict(pi=[128,128,128], vf=[128,128,128])],
    #   activation_fn=ReLU,
    )
#policy_kwargs=None

if __name__ == "__main__":
    train = True

    # Parameter für Self-Play
    total_generations = 80  # Anzahl der Self-Play-Generationen
    timesteps_per_generation = int(5e5)  # Trainingsschritte pro Generation

    if train:
        # Algorithmus-Auswahl
        algorithm_choice = "PPO_mask"

        # Timer starten
        start = time.time()

        # Initialisiere die Umgebung ohne Gegner (wird später gesetzt)
        #env = RPSEnv(render_mode=None, opponent=None, size=4, n_holes=2)

        model = MaskablePPO.load(f"model_Generation_{30}_a")
        env = RPSEnv(render_mode=None, opponent=RandomOpponent(), size=4, n_holes=2)

        env = ActionMasker(env, mask_fn)

        # Initialisiere das Modell
        if algorithm_choice == "PPO_mask":
            model = MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs, learning_rate=3e-4, gamma=0.99, verbose=1) #stats_window_size
            #model = MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs, learning_rate=1e-4, gamma=0.9, verbose=1) #stats_window_size
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm_choice}")
        #model = MaskablePPO.load(f"model_Generation_last")

        # print(model.policy)

        # Modell des Gegners initialisieren (in der ersten Generation wird der Base oponent verwendet

        #opponent = BaseOpponent()

        for generation in range(total_generations):
            print(f"=== Generation {generation + 1}/{total_generations} ===")
            # Setze den Gegner in der Umgebung

            #ic(model.env.get_attr("opponent"))
            #ic(model.env.get_attr("env")[0].env.opponent)

            #print(model.env.opponent)

            # Trainiere das Modell für diese Generation
            model.learn(total_timesteps=timesteps_per_generation, log_interval=int(5e1))
            #ic(model.env.get_attr("opponent"))
            #ic(model.env.get_attr("env")[0].env.opponent)

            # Aktualisiere das Gegner-Modell am Ende der Generation

            #opponent_model = copy.deepcopy(model)  # Speichere eine Kopie des aktuellen Modells als neuen Gegner
            model.save(f"model_Generation_{generation + 1}")

            opponent_model = MaskablePPO.load(f"model_Generation_{generation + 1}")
            # reload the model compleatly detaches the opponent_model from model
            #model.env.opponent = ModelOpponent(opponent_model)  # Setze den neuen Gegner
            #env.env_method("set_mu", next(mus))

            op = ModelOpponent(opponent_model)
            #model.env.set_attr("opponent", op ) # does not do the right thing, because model.env is the vectorized env contaning the env(s)
            #for e in model.env.get_attr("env"): # for every env e in the vectorized env model.env:
            #    e.env.opponent = op             # this would work
            model.env.env_method("set_opponent",op) # more elegant way

            #ic(model.env.get_attr("env")[0].env.opponent)


            print("total elapsed time:", time.time() - start)

        # Speichere das finale Modell
        model.save(f"{algorithm_choice.lower()}_rps_final")
        env.close()
        del env

        # Timer stoppen
        end = time.time()
        print(f"Training abgeschlossen in {end - start:.2f} Sekunden.")

        # Test der letzte Modellversion
        final_opponent = ModelOpponent(opponent_model)
        env = RPSEnv(render_mode=None, opponent=None, size=4, n_holes=2)
        wins, looses = evaluation(env, agent_model = model, n_epochs=int(2e2))
        print(f"result: wins: {wins}, looses: {looses}")
        env.close()


    opponent = None
    env = RPSEnv(render_mode="human", opponent=RandomOpponent(), size=4, n_holes=2)
    """
    player = MaskablePPO.load(f"model_Generation_{20}")
    wins, looses = evaluation(env, agent_model=player, n_epochs=int(1e3))
    """

    def eval(generation):
        #print(f"=== Generation {generation}/{total_generations} ===")
        player = MaskablePPO.load(f"model_Generation_{generation}")
        start = time.time()
        wins, looses = evaluation(env, agent_model=player, n_epochs=int(1e3))
        t = time.time() - start
        print(f"gen: {generation}, result: wins: {wins}, looses: {looses}, time: {t}")
        return generation, wins, looses

    """
    parallel(n_jobs=8)(
        delayed(eval)(gen) for gen in range(1, total_generations)
    )
    """

    for generation in range(total_generations):
        #for i in range(3):
            print(f"=== Generation {generation + 1}/{total_generations} ===")
            player = MaskablePPO.load(f"model_Generation_{generation +1}")
            start = time.time()
            wins, looses = evaluation(env, agent_model=player, n_epochs=int(1e4))
            print(time.time() - start)
            print(f"result: wins: {wins}, looses: {looses}")

    env.close()


    env = RPSEnv(render_mode=None, opponent=None, size=4, n_holes=2)
    for player in range(20, total_generations):
        for oponent in range(20, total_generations):
        #for i in range(3):
            print(f"=== {player} vs. {oponent} ===")
            player_agent = MaskablePPO.load(f"model_Generation_{player}")
            opponent_agent = MaskablePPO.load(f"model_Generation_{player}")
            env.opponent = ModelOpponent(opponent_agent)
            wins, looses = evaluation(env, agent_model=player_agent, n_epochs=int(1e1))
            print(f"result: wins: {wins}, looses: {looses}")

    env.close()




