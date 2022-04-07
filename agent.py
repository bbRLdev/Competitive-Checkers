from game import Game
import numpy as np

class Agent(): 
    def __init__(self, color) -> None:
        self.color = color
        pass

    def get_action(self, game) -> int:
        pass

    def get_piece(self) -> int:
        return self.color

class RandomAgent(Agent):
    def __init__(self, color) -> None:
        super().__init__(color)

    def get_action(self, game) -> int:
        moves = game.get_valid_moves()
        return np.random.choice(moves)

class FeatureAgent(Agent):
    def __init__(self, color) -> None:
        super().__init__(color)
        self.feature_functions = [block_cols(), win_cols(), opp_pieces_per_col(), my_pieces_per_col()]

    def get_action(self, game) -> int:
        vect = self.state_to_vect(game.board)

    def state_to_vect(self, state):
        state_vect = []
        for feat_func  in self.feature_functions:
            state_vect += feat_func(state)
    
def block_cols(board):
    pass

def win_cols(board):
    pass

def opp_pieces_per_col(board, opp_color):
    ret = []
    print(board.shape())
    rows, cols = board.shape() #get dimensions of board
    np.where(board == opp_color, board, True)
    for c in range(cols):
        count = 0
        for r in range(rows):
            if board[r, c]:
                bool = False
        #use numpy.where

    return result
        
def my_pieces_per_col(board, my_color):
    pass#same as above