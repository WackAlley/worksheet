import argparse
from stable_baselines3 import A2C, DQN, PPO
from triton.profiler import start

from rps_game.env import RPSEnv
from utilities.RPSEnvWrapper import RPSEnvWrapper
import time


#if __name__ == "__main__":

def main(algorithm_choice):
    start = time.time()
    env = RPSEnv(render_mode=None, opponent=None, size=4, n_holes=2)
    # Wrappen der Umgebung, bei übergabe der nicht erlaubten aktionen, wird es mit negativen reward
    # bestraft und zufällige aktion gemacht anstadt Fehler zu werfen
    wrapped_env = RPSEnvWrapper(env)

    # Wähle den gewünschten Algorithmus (DQN, A2C oder PPO)
    #algorithm_choice = "PPO"

    if algorithm_choice == "A2C":
        model = A2C("MlpPolicy",wrapped_env, learning_rate=7e-4 ,gamma=0.99, verbose=1)
    elif algorithm_choice == "DQN":
        model = DQN("MlpPolicy",wrapped_env, learning_rate=1e-4 ,gamma=0.99, verbose=1)
    elif algorithm_choice == "PPO":
        model = PPO("MlpPolicy", wrapped_env, learning_rate=3e-4 ,gamma=0.99,verbose=1)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_choice}")

    # Trainiere das Modell
    model.learn(total_timesteps=int(5e6), log_interval=4) #1e5
    model.save(f"{algorithm_choice.lower()}_rps")  # Speichern des Modells
    wrapped_env.close()
    del env, wrapped_env

    end = time.time()
    print(end-start)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start training with a selected algorithm.")
    parser.add_argument(
        "algorithm", choices=["A2C", "DQN", "PPO"], help="The algorithm to use (A2C, DQN, PPO)"
    )
    args = parser.parse_args()

    main(args.algorithm)
