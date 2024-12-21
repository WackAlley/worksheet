from q_table import QTable
from rps_game.env import RPSEnv
from utilities.player_models import QTableAgent_lookup_only, QTableAgent_with_exploration, MyOponentModel

from rps_game.opponent import ModelOpponent
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO



def evaluation(env,agent_model, n_epochs):
    wins = 0
    looses = 0
    for epoch in range(n_epochs):
        observation_this_state, info = env.reset()  # Startzustand
        done = False
        while not done:
            action_mask = env.compute_action_mask()  # Gültige Aktionen
            action, _ = agent_model.predict(observation_this_state, action_masks=action_mask)
            observation_next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                if reward == -1:
                    looses += 1
                if reward >= +0.15:
                    wins += 1
            observation_this_state = observation_next_state
    env.reset()
    return wins, looses

def training(env,agent_model, n_epochs):
    for epoch in range(n_epochs):
        observation_this_state, info = env.reset()  # Startzustand
        done = False
        while not done:
            action_mask = env.compute_action_mask()  # Gültige Aktionen
            action, _ = agent_model.predict(observation_this_state, action_masks=action_mask)
            observation_next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if not done:
                action_mask_next_state = env.compute_action_mask()
            else:
                action_mask_next_state = None
            agent_model.learn(observation_this_state, action, reward, observation_next_state, action_mask_next_state)
            observation_this_state = observation_next_state
    env.reset()


if __name__ == "__main__":
        # example training:
        learning_rate = 0.8
        discount_factor = 0.95
        exploration_prob = 0.2
        n_epochs = int(1e4)
        qtable = QTable(learning_rate=learning_rate, discount_factor=discount_factor, exploration_prob=exploration_prob)

        #train against self, player and opponend have same q_table:
        model_self = QTableAgent_with_exploration(qtable, exploration_prob)
        model_opponent = QTableAgent_lookup_only(qtable)
        player_opponent = MyOponentModel(model_opponent)
        env = RPSEnv(render_mode=None, opponent=player_opponent, size=4, n_holes=2)
        training(env,model_self, n_epochs)
        env.close()
        # Speichern
        qtable.save("qtable.pkl")

        del env, qtable, model_self, model_opponent, player_opponent, n_epochs

        # example evaluation:
        n_epochs = int(1e3)
        qtable = QTable.load("qtable.pkl")
        agent_model = QTableAgent_with_exploration(qtable, exploration_prob)
        env = RPSEnv(render_mode=None, opponent=None, size=4, n_holes=2)
        #opponent = None -> Base Opponent with random actions
        wins, looses = evaluation(env, agent_model, n_epochs)
        print(f"result: wins: {wins}, looses: {looses}")
        env.close()


