
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import time
from rps_game.env import RPSEnv  # Deine benutzerdefinierte Umgebung
from utilities.RPSEnvWrapper import RPSEnvWrapper


if __name__ == "__main__":


    # Erstelle die RPSEnv-Umgebung
    env = RPSEnv(render_mode=None)
    # Wrappen der Umgebung
    wrapped_env = RPSEnvWrapper(env)



    model = DQN("MlpPolicy", wrapped_env, verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)
    model.save("dqn_rps")


    #del model  # remove to demonstrate saving and loading
    wrapped_env.close()
    del env, wrapped_env
    model = DQN.load("dqn_rps")
    env = RPSEnv(render_mode="human")

    wrapped_env = RPSEnvWrapper(env)
    obs, info = wrapped_env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        if terminated or truncated:
            obs, info = wrapped_env.reset()


