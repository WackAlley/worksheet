import gymnasium as gym
import numpy as np

class RPSEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def step(self, action: int):
        """
        Wenn eine ungültige Aktion ausgewählt wird, wird sie bestraft.
        Eine zufällige gültige Aktion wird stattdessen ausgeführt.
        """
        # Berechne die Maskierung der gültigen Aktionen
        mask = self.env.compute_action_mask()
        # Wenn die gewählte Aktion ungültig ist
        if not mask[action]:
            # Wähle eine zufällige gültige Aktion aus
            valid_actions = np.where(mask)[0]
            action = np.random.choice(valid_actions)
            observation, reward, terminated, truncated, info = self.env.step(action)
            # Strafte für die ungültige Aktion

            reward  = -0.8
            return observation, reward, terminated, truncated, info

        # Führe den Schritt aus, wenn die Aktion gültig ist
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

