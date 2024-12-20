import numpy as np
from numpy.typing import NDArray
from enum import Enum
from dataclasses import dataclass, field
from typing import ClassVar, Optional


class Piece(Enum):
    ROCK = 1
    PAPER = 2
    SCISSORS = 3

    @staticmethod
    def new_from_board_encoding(board_enc: int):
        # Board encoding for player i is i + (normal encoding)
        return Piece((board_enc-4) % 3 + 1)
    
    @staticmethod
    def owner_id_from_board_encoding(board_enc: int) -> int:
        return (board_enc - 4) // 3 + 1

    def cycle(self):
        return Piece(self.value % 3 + 1)
    
    def fightable(self, opponent: "Piece"):
        return self.value != opponent.value
    
    def fight(self, opponent: "Piece"):
        """
        Performs a RPS fight.

        Returns
            1 if self won, -1 if opponent won, 0 if tie
        """
        p, o = self.value, opponent.value
        return (p - o + 4) % 3 - 1


class Obstacle(Enum):
    HOLE = -2
    GOAL = -100


class Action(Enum):
    MOVE_RIGHT = 0
    MOVE_UP = 1
    MOVE_LEFT = 2
    MOVE_DOWN = 3
    CONVERT = 4

    def displace_vec(self):
        displace_vecs = {
            Action.MOVE_RIGHT: np.array([0, 1]),
            Action.MOVE_UP: np.array([-1, 0]),
            Action.MOVE_LEFT: np.array([0, -1]),
            Action.MOVE_DOWN: np.array([1, 0])
        }
        return displace_vecs[self]
    
    def is_move(self):
        moves = [Action.MOVE_RIGHT, Action.MOVE_UP, Action.MOVE_LEFT, Action.MOVE_DOWN]
        return self in moves
    
    def is_conversion(self):
        return self == Action.CONVERT


@dataclass
class Player():
    pos: NDArray[np.int8]
    piece: Piece
    id: int = field(init=False)

    _next_id: ClassVar[int] = 0 # Enumerates players

    def __post_init__(self):
        self.id = Player._next_id
        Player._next_id += 1

    def get_translate_pos(self, move_action: Action) -> NDArray[np.int8]:
        assert move_action.is_move()
        return self.pos + move_action.displace_vec()

    def move(self, move_action: Action) -> NDArray[np.int8]:
        new_pos = self.get_translate_pos(move_action)
        self.pos = new_pos
        return new_pos
    
    def convert(self) -> Piece:
        new_type = self.piece.cycle()
        self.piece = new_type
        return new_type
    
    def act(self, action: Action) -> None:
        if action.is_conversion():
            self.convert()
        elif action.is_move():
            self.move(action)
        else:
            raise ValueError(f"Invalid action type {action}")
    

@dataclass
class Hole():
    pos: NDArray[np.int8]


@dataclass
class Barrier():
    pass



class Game():
    """
    Game class
    """    
    def __init__(self,
                 size: int = 4,
                 *,
                 n_players: int = 2,
                 n_holes: int = 0,
                 rng: np.random.Generator = None):
        assert size > 1
        self.size = size
        assert n_players > 1
        if n_players != 2:
            raise NotImplementedError()
        self.n_players = n_players
        self.n_holes = n_holes
        self._static_hole_config: NDArray = None
        if not rng:
            rng = np.random.default_rng()
        self.rng = rng
        self.initialize_game_random()

        # Who's turn it is
        self.next_player = self.players[0].id
    
    @property
    def board(self) -> NDArray[np.int8]:
        """
        Encoding
            0 <=> empty
            1/2/3 + i*3 <=> rock/paper/scissors of player i
            -2 <=> hole

        Returns
            Board as numpy array.
        """
        if self._board_cache is None:
            self._board_cache = self._generate_board()
        return self._board_cache
    
    def _generate_board(self) -> NDArray[np.int8]:
        board = np.zeros((self.size, self.size), dtype=np.int8)
        for player in self.players:
            piece_encoding = player.piece.value + player.id * 3
            board[tuple(player.pos)] = piece_encoding
        for hole in self.holes:
            board[tuple(hole.pos)] = Obstacle.HOLE.value
        return board
    
    def set_static_hole_config(self, positions: NDArray) -> None:
        # Holes that persist a reset
        assert len(positions) == self.n_holes
        self._static_hole_config = positions
        self.holes = [Hole(positions[i]) for i in range(self.n_holes)]
        self.initialize_game_random()

    def initialize_game_random(self) -> None:
        self.players = []
        if self._static_hole_config is None:
            self.holes = []
        self._board_cache = None
        empty_cells = np.argwhere(self.board == 0)
        sampled_pos = self.rng.choice(empty_cells,
                                      size=self.n_players+self.n_holes,
                                      replace=False)
        sampled_types = self.rng.choice(Piece, size=self.n_players)
        self.players = [Player(sampled_pos[i], sampled_types[i])
                        for i in range(self.n_players)]
        for i, player in enumerate(self.players): player.id = i
        self.next_player = self.players[0].id
        if self._static_hole_config is None:
            self.holes = [Hole(sampled_pos[i+self.n_players])
                          for i in range(self.n_holes)]
        self._board_cache = None
    
    reset = initialize_game_random # alias
    
    def in_bounds(self, x, y):
        return (0 <= x < self.size) and (0 <= y < self.size)
    
    def is_game_over(self) -> bool:
        """
        Game over when one player dead.

        Returns
            if game is over, ID of winning player
        """
        return len(self.players) == 1
    
    def get_winner_id(self) -> int:
        if self.is_game_over():
            return self.players[0].id
    
    def is_action_valid(self, action: Action) -> bool:
        """
        Game rules:
            - No moving outside the grid
            - Move to location where other piece is only iff fightable
            - Conversion only if no other piece in neighborhood
        """
        player: Player = self.players[self.next_player]
        if action.is_move():
            new_pos = player.get_translate_pos(action)
            out_of_bounds = not self.in_bounds(*new_pos)
            if out_of_bounds:
                return False
            occupied = self.board[tuple(new_pos)] > 0
            if occupied:
                occupied_by = Piece.new_from_board_encoding(self.board[tuple(new_pos)])
                not_fightable = not player.piece.fightable(occupied_by)
                if not_fightable:
                    return False
            return True
        if action.is_conversion():
            surrounded = False
            directions = [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])]
            for dir in directions:
                check = player.pos + dir
                if not self.in_bounds(*check):
                    continue
                surrounded = surrounded or (self.board[tuple(check)] > 0)
            if surrounded:
                return False
            return True
        else:
            raise ValueError(f"Invalid action type {action}")
        
    def act(self, action: Optional[Action]) -> None:
        if action is None:
            self.next_player = (self.next_player + 1) % self.n_players
            return
        if not self.is_action_valid(action):
            raise ValueError(f"Action {action} not allowed")
        if self.is_game_over():
            raise Warning("Recorded action in terminated game")
        player: Player = self.players[self.next_player]
        player.act(action)
        if action.is_move():
            met_entity_encoded: np.int8 = self.board[tuple(player.pos)]
            # Fell in hole
            if met_entity_encoded == Obstacle.HOLE.value:
                del self.players[self.next_player]
            # Fight
            elif met_entity_encoded > 0:
                met_piece = Piece.new_from_board_encoding(met_entity_encoded)
                outcome = player.piece.fight(met_piece)
                if outcome > 0: # player won
                    looser_id = Piece.owner_id_from_board_encoding(met_entity_encoded)
                    del self.players[looser_id]
                elif outcome < 0: # player lost
                    del self.players[self.next_player]
        
        self.next_player = (self.next_player + 1) % self.n_players
        self._board_cache = None
    
    def __str__(self):
        out = ""

        def fmt_encoded_piece(enc: int):
            if enc == 0:
                return " "
            elif enc == 1:
                return "r"
            elif enc == 4:
                return "R"
            elif enc == 2:
                return "p"
            elif enc == 5:
                return "P"
            elif enc == 3:
                return "s"
            elif enc == 6:
                return "S"
            elif enc == -2:
                return "O"
        
        out += "-" * (self.size+2) + "\n"
        for i in range(self.size):
            out += "|"
            for j in range(self.size):
                out += fmt_encoded_piece(self.board[i, j])
            out += "|\n"
        out += "-" * (self.size+2)
        return out
    