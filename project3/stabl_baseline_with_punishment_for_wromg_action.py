from stable_baselines3 import A2C, DQN, PPO

from rps_game.env import RPSEnv
from utilities.RPSEnvWrapper import RPSEnvWrapper
import time


if __name__ == "__main__":


    # Wähle den gewünschten Algorithmus (DQN, A2C oder PPO)
    algorithm_choice = "PPO"

    start = time.time()
    env = RPSEnv(render_mode=None, opponent=None, size=4, n_holes=2)
    # Wrappen der Umgebung, bei übergabe der nicht erlaubten aktionen, wird es mit negativen reward
    # bestraft und zufällige aktion gemacht anstadt Fehler zu werfen
    wrapped_env = RPSEnvWrapper(env)



    if algorithm_choice == "A2C":
        model = A2C("MlpPolicy",wrapped_env, learning_rate=7e-4 ,gamma=0.99, verbose=1)
    elif algorithm_choice == "DQN":
        model = DQN("MlpPolicy",wrapped_env, learning_rate=1e-4 ,gamma=0.99, verbose=1)
    elif algorithm_choice == "PPO":
        model = PPO("MlpPolicy", wrapped_env, learning_rate=3e-4 ,gamma=0.99,verbose=1)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_choice}")

    # Trainiere das Modell
    model.learn(total_timesteps=int(1e5), log_interval=4) #1e5
    model.save(f"{algorithm_choice.lower()}_rps")  # Speichern des Modells
    wrapped_env.close()
    del env, wrapped_env

    end = time.time()
    print(end-start)

    # Modell testen
    """
    model = model.load(f"{algorithm_choice.lower()}_rps")  # Lade das Modell

    env = RPSEnv(render_mode="human", opponent=None, size=4, n_holes=2)
    wrapped_env = RPSEnvWrapper(env)
    obs, info = wrapped_env.reset()

    while True:
        # Wähle eine Aktion basierend auf der Vorhersage des Modells
        action, _states = model.predict(obs, deterministic=True)

        # Führe die Aktion aus und erhalte die nächste Beobachtung
        obs, reward, terminated, truncated, info = wrapped_env.step(action)

        if terminated or truncated:
            obs, info = wrapped_env.reset()
    """


