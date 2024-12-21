from rps_game.env import RPSEnv
import numpy as np
from rps_game.opponent import BaseOpponent
from numpy.typing import NDArray
import gymnasium as gym
from rps_game.game import Action


class QTableAgent_lookup_only():
    #uses the provided q table to look up the best move
    def __init__(self, qtable):
        self.qtable = qtable

    def predict(self, observation, action_masks):
        valid_actions = np.argwhere(action_masks).flatten()
        valid_q_values = self.qtable.get_all_q_given_state(observation, valid_actions)
        return valid_actions[np.argmax(valid_q_values)], None

class QTableAgent_with_exploration():
    def __init__(self, qtable, exploration_prob):
        self.qtable = qtable
        self.exploration_prob = exploration_prob

    def predict(self, observation, action_masks):
        valid_actions = np.argwhere(action_masks).flatten()

        if np.random.rand() < self.exploration_prob:
            return np.random.choice(valid_actions), None
        else:
            valid_q_values = self.qtable.get_all_q_given_state(observation, valid_actions)
            return valid_actions[np.argmax(valid_q_values)], None
    def learn(self, observation_this_state, action, reward, observation_next_state, action_mask_next_state):
        #next stete is terminal is indeicated by action_mask_next_State = None
        if action_mask_next_state is not None:
            valid_actions_next_state = np.argwhere(action_mask_next_state).flatten()
        else:
            valid_actions_next_state = tuple("no actions")
        self.qtable.update_table(observation_this_state, action, observation_next_state, valid_actions_next_state, reward)
    # ic("updated")




class MyOponentModel(BaseOpponent):
    """
    Performs actions as dicated by the model.
    The model must implement a predict method.
    Observations are switched, such that the observation
    is always of the perspective of player 0.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

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
            obs = MyOponentModel.switch_observation_encoding(obs)
        action_mask = env.compute_action_mask()
        action_id, _ = self.model.predict(obs, action_mask)
        action = Action(action_id)
        env.game.act(action)
