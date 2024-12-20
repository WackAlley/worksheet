import os
import gymnasium as gym
import pygame
import numpy as np
from numpy.typing import NDArray

from rps_game.game import Game, Player, Action, Obstacle, Piece
from rps_game.opponent import BaseOpponent


STATIC_HOLE_CONFIG = np.array([(1, 1), (2, 3)])


class RPSEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "console"],
        "render_fps": 4
    }

    def __init__(self, render_mode=None, *,
                 opponent: BaseOpponent=None,
                 static_hole_config=True,
                 **game_kwargs):
        self.game = Game(**game_kwargs)
        self.opponent = opponent if opponent is not None else BaseOpponent()

        if self.game.n_holes == 2 and static_hole_config:
            static_hole_config = STATIC_HOLE_CONFIG
            self.game.set_static_hole_config(static_hole_config)

        """
        Observation space:
        ------------------
        size x size,
        Elements are 0-empty, 1/4-rock, 2/5-paper, 3/6-scissors
        for players 0/1
        """
        self.observation_space = gym.spaces.Box(
            low=0, high=6,
            shape=(self.game.size, self.game.size), dtype=np.int8
        )

        """
        Action space:
        -------------
        An action is one player moving one piece or converting a piece cyclic.
        Direction moves are 0-3, 4 is conversion
        """
        self.action_space = gym.spaces.Discrete(5)

        """
        Rendering
        """
        self.window_size = 512 # The size of the PyGame window
        self.window = None
        self.clock = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if render_mode == "human":
            self._load_render_assets()

        # Timer
        self.t = 0

    def _get_obs(self):
        return self.game.board.copy()

    def _get_info(self):
        return {"time": self.t}

    def reset(self, seed=None, options=None):
        # Seed self.np_random
        super().reset(seed=seed)
        self.game.rng = self.np_random
        self.game.reset()
        self.t = 0
        observation = self._get_obs()
        info = self._get_info()
        self.render()
        return observation, info

    def compute_action_mask(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        for action in np.arange(self.action_space.n):
            mask[action] = self.game.is_action_valid(Action(action))
        if not np.any(mask): # no possible moves
            raise EnvironmentError("No move possible.")
        return mask

    def step(self, action: int):
        """
        Action must be element of action_space.
        """
        action: Action = Action(action)

        # Act
        goal_hit = self.game.act(action)
        self.render()
        # Other player response (is player 1 if not BaseOpponent)
        self.opponent.respond(self, self.game.rng)
        self.render()

        terminated = self.game.is_game_over()
        # Compute reward
        reward = 0
        if terminated:
            if goal_hit:
                reward += 0.2
            else:
                winner = self.game.get_winner_id()
                reward += 1 - winner * 2 # +1 for win of player 0, -1 for loss
        else:
            reward -= 0.05 # time penalty
        observation = self._get_obs()
        info = self._get_info()
        self.t += 1
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()
        elif self.render_mode == "console":
            return self._render_console()

    def _render_console(self):
        print(self.game.board)

    def _load_render_assets(self):
        asset_base = "rps_game/assets"
        cell_size = self.window_size // self.game.size
        self.images = {
            Piece.ROCK.value: pygame.image.load(os.path.join(asset_base, "rock.png")),
            Piece.PAPER.value: pygame.image.load(os.path.join(asset_base, "paper.png")),
            Piece.SCISSORS.value: pygame.image.load(os.path.join(asset_base, "scissors.png")),
            Obstacle.HOLE.value: pygame.image.load(os.path.join(asset_base, "hole.png")),
            Obstacle.GOAL.value: pygame.image.load(os.path.join(asset_base, "goal.png"))
        }
        for key, image in self.images.items():
            self.images[key] = pygame.transform.scale(image, (cell_size // 2, cell_size // 2))

    def _render_frame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            # Increase window width for scoreboard
            self.window = pygame.display.set_mode((self.window_size + 100, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()
        colors = [(31, 119, 180), (214, 39, 40)]  # tab:blue and tab:red
        hole_color = (93, 93, 93)
        bg_color = (255, 255, 255)  # White background
        grid_color = (200, 200, 200)  # Light gray grid
        cell_size = self.window_size // self.game.size
        self.window.fill(bg_color)
        # Draw grid
        for x in range(self.game.size + 1):
            pygame.draw.line(self.window, grid_color, (x * cell_size, 0), (x * cell_size, self.window_size))
            pygame.draw.line(self.window, grid_color, (0, x * cell_size), (self.window_size, x * cell_size))
        # Draw pieces
        for i, player in enumerate(self.game.players):
            x, y = player.pos
            center = (y * cell_size + cell_size // 2, x * cell_size + cell_size // 2)
            pygame.draw.circle(self.window, colors[i], center, cell_size // 3)
            image = self.images[player.piece.value]
            image_rect = image.get_rect(center=center)
            self.window.blit(image, image_rect)
        for hole in self.game.holes:
            x, y = hole.pos
            center = (y * cell_size + cell_size // 2, x * cell_size + cell_size // 2)
            pygame.draw.circle(self.window, hole_color, center, cell_size // 3)
            image = self.images[Obstacle.HOLE.value]
            image_rect = image.get_rect(center=center)
            self.window.blit(image, image_rect)
        if self.game.spawn_goal_field:
            x, y = self.game.goal.pos
            center = (y * cell_size + cell_size // 2, x * cell_size + cell_size // 2)
            image = self.images[Obstacle.GOAL.value]
            image_rect = image.get_rect(center=center)
            self.window.blit(image, image_rect)
        # Legend
        legend_x = self.window_size + 5
        font = pygame.font.SysFont(None, 24)
        legend_title = font.render("Legend:", True, (50, 50, 50))
        self.window.blit(legend_title, (legend_x, 20))
        for player in range(2):
            player_color = colors[player]
            player_text = font.render(f"Player {player}", True, player_color)
            self.window.blit(player_text, (legend_x, 50 + player * 20))
        # Update display and manage frame rate
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        if self.game.is_game_over():
            print(f"Game over! Player {self.game.get_winner_id()} won!")
        # Handle events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
