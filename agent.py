from game import Game
import numpy as np

class Agent(): 
    def __init__(self) -> None:
        pass

    def get_action(self, game) -> int:
        pass

class RandomAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def get_action(self, game) -> int:
        moves = game.get_valid_moves()
        return np.random.choice(moves)
