import numpy as np
import time
from rps_game.env import RPSEnv


if __name__ == "__main__":
    env = RPSEnv(render_mode="human")
    observation, info = env.reset(seed=42)

    for _ in range(100):
        mask = env.compute_action_mask()
        valid_actions = np.argwhere(mask)
        action = valid_actions[0]
        observation, reward, terminated, truncated, info = env.step(action)
        time.sleep(1)
        
    env.close()
