from rps_game.env import RPSEnv
import time

import gymnasium as gym
import numpy as np
from rps_game.opponent import ModelOpponent

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from utilities.training_and_evaluation import evaluation


def mask_fn(env: gym.Env) -> np.ndarray:
    return env.compute_action_mask()


if __name__ == "__main__":

    # Wähle den gewünschten Algorithmus (DQN, A2C oder PPO)
    algorithm_choice = "PPO_mask"

    start = time.time()
    env = RPSEnv(render_mode=None, opponent=None, size=4, n_holes=2)
    env = ActionMasker(env, mask_fn)  # Wrap to enable masking

    # habe bisher nur den einen gefunden, der mit maskable policies arbeitet
    if algorithm_choice == "PPO_mask":
        model = MaskablePPO("MlpPolicy", env, learning_rate=3e-4 ,gamma=0.99,verbose=1)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_choice}")

    # Trainiere das Modell
    model.learn(total_timesteps=int(1e3), log_interval=4) #1e5
    model.save(f"{algorithm_choice.lower()}_rps")  # Speichern des Modells
    env.close()
    del env

    end = time.time()
    print(end-start)

    # Modell testen

    model = model.load(f"{algorithm_choice.lower()}_rps")  # Lade das
    # test gegen sich selber
    opponent = ModelOpponent(model)
    env = RPSEnv(render_mode="human", opponent=opponent, size=4, n_holes=2)
    obs, info = env.reset()


    wins, looses = evaluation(env, agent_model = model, n_epochs=int(1e3))
    print(f"result: wins: {wins}, looses: {looses}")
    env.close()




