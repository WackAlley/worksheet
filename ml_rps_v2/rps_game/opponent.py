import numpy as np
import gymnasium as gym
from numpy.typing import NDArray
from sb3_contrib import MaskablePPO

from rps_game.game import Action


class BaseOpponent:
    """
    Does absolutely nothing.
    """
    def respond(self, env: gym.Env, rng: np.random.Generator) -> None:
        _ = rng, env
        return


class PassiveOpponent(BaseOpponent):
    """
    Performs an empty action.
    """
    def respond(self, env: gym.Env, rng: np.random.Generator):
        _ = rng
        if env.game.is_game_over():
            return
        env.game.act(None)


class RandomOpponent(BaseOpponent):
    """
    Performs a random action.
    """
    def respond(self, env: gym.Env, rng: np.random.Generator):
        if env.game.is_game_over():
            return
        actions = list(Action)
        rng.shuffle(actions)
        for action in actions:
            if env.game.is_action_valid(action):
                env.game.act(action)
                return
        # if all actions invalid, do nothing
        env.game.act(None)
    

class ModelOpponent(BaseOpponent):
    """
    Performs actions as dicated by the model.
    The model must implement a predict method.
    Observations are switched, such that the observation
    is always of the perspective of player 0.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.maskable = isinstance(model, MaskablePPO)
    
    @staticmethod
    def switch_observation_encoding(obs: NDArray):
        """
        Essentially switches the player ID's, returning the observation
        with player 0's piece encoding being 4/5/6 and player 1's piece
        encoding being 1/2/3.
        """
        obs_dup = obs.copy()
        obs_dup[(obs == 1) | (obs == 2) | (obs == 3)] += 3
        obs_dup[(obs == 4) | (obs == 5) | (obs == 6)] -= 3
        return obs_dup
    
    def respond(self, env: gym.Env, rng: np.random.Generator):
        if env.game.is_game_over():
            return
        obs = env._get_obs()
        # Ensure that model always sees observation of player 0 perspective
        if env.game.next_player == 1:
            obs = ModelOpponent.switch_observation_encoding(obs)
        if self.maskable:
            am = env.compute_action_mask()
            action_id, _ = self.model.predict(obs, action_masks=am)
        else:
            action_id, _ = self.model.predict(obs)
        action = Action(action_id)
        env.game.act(action)
