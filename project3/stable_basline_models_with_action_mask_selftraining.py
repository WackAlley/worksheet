from rps_game.env import RPSEnv
import time
import gymnasium as gym
import numpy as np
from rps_game.opponent import ModelOpponent
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from utilities.training_and_evaluation import evaluation
from rps_game.opponent import BaseOpponent


def mask_fn(env: gym.Env) -> np.ndarray:
    """
    Generiert eine Aktionsmaske basierend auf der Umgebung.
    """
    return env.compute_action_mask()


if __name__ == "__main__":
    # Algorithmus-Auswahl
    algorithm_choice = "PPO_mask"

    # Timer starten
    start = time.time()

    # Initialisiere die Umgebung ohne Gegner (wird später gesetzt)
    env = RPSEnv(render_mode=None, opponent=None, size=4, n_holes=2)
    env = ActionMasker(env, mask_fn)

    # Initialisiere das Modell
    if algorithm_choice == "PPO_mask":
        model = MaskablePPO("MlpPolicy", env, learning_rate=3e-4, gamma=0.99, verbose=1)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_choice}")

    # Parameter für Self-Play
    total_generations = 5  # Anzahl der Self-Play-Generationen
    timesteps_per_generation = int(1e2)  # Trainingsschritte pro Generation

    # Modell des Gegners initialisieren (in der ersten Generation wird der Base oponent verwendet

    opponent = BaseOpponent()

    for generation in range(total_generations):
        print(f"=== Generation {generation + 1}/{total_generations} ===")

        # Setze den Gegner in der Umgebung
        env.opponent = opponent

        # Trainiere das Modell für diese Generation
        model.learn(total_timesteps=timesteps_per_generation, log_interval=4)

        # Aktualisiere das Gegner-Modell am Ende der Generation

        #opponent_model = copy.deepcopy(model)  # Speichere eine Kopie des aktuellen Modells als neuen Gegner
        model.save(f"model_Generation_{generation + 1}")
        opponent_model = MaskablePPO.load(f"model_Generation_{generation + 1}")
        # reload the model compleatly detaches the opponent_model from model
        opponent = ModelOpponent(opponent_model)  # Setze den neuen Gegner

    # Speichere das finale Modell
    model.save(f"{algorithm_choice.lower()}_rps_final")
    env.close()
    del env

    # Timer stoppen
    end = time.time()
    print(f"Training abgeschlossen in {end - start:.2f} Sekunden.")

    # Test gegen die letzte Modellversion
    final_opponent = ModelOpponent(opponent_model)
    env = RPSEnv(render_mode=None, opponent=final_opponent, size=4, n_holes=2)
    wins, looses = evaluation(env, agent_model = model, n_epochs=int(1e4))
    print(f"result: wins: {wins}, looses: {looses}")
    env.close()



