from game import Game
import numpy as np
import math
from scipy.signal import convolve2d 

FEAT_EXTRACTOR_LENS = [7,7,7,7,7]
FEAT_EXTRACTOR_LENS_2 = [42]

class Agent(): 
    def __init__(self, color) -> None:
        self.color = color
        self.opponent_color = (color % 2) + 1
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

class HumanAgent(Agent):
    def __init__(self, color) -> None:
        super().__init__(color)
    
    def get_action(self, game : Game) -> int:
        input_action = input()
        action = int(input_action)
        while action < 0 or action > 6 or action not in game.get_valid_moves():
            print("Select a valid piece placement from 0-6")
            input_action = input()
            action = int(input_action)

        return action

class SARSA_FeatureAgent(Agent):
    def __init__(self, color, nA = 7, w=None) -> None:
        super().__init__(color)
        self.feature_functions = [self.opp_win, self.my_win, self.losing_moves,  self.opp_pieces_per_col, self.my_pieces_per_col]
        self.num_actions = nA

    def get_raw_board(self, game: Game):
        copy = np.copy(game.board)
        return np.ravel(copy)
    
    def get_action(self, game : Game) -> int:
        if len(game.get_valid_moves()) == 0:
            return None
        Q = [np.dot(self.w, self(game, False, a)) for a in range(self.num_actions)]
        act = np.argmax(Q)
        while not game.is_valid_location(act):
            Q[act] = np.NINF
            act = np.argmax(Q)
        return act

    def set_weights(self, weights):
        self.w = weights

    def feature_vector_len(self):
        # return self.num_actions * len(self.feature_functions)
        return np.sum(FEAT_EXTRACTOR_LENS)

    def __call__(self, game, done, action):
        if(done):
            return np.zeros(self.num_actions * self.feature_vector_len())
        x = self.get_state_action_vector(game, action)
        return x

    def get_state_action_vector(self, game : Game, action):
        vals = []
        for a in range(self.num_actions):
            if action == a:
                vals.extend(self.state_to_vect(game))
            else:
                vals.extend([0 for _ in range(self.feature_vector_len())])
        vec = np.array(vals, dtype=np.float32)
        return vec

    def state_to_vect(self, game):
        state_vect = []
        for feat_func  in self.feature_functions:
            state_vect.extend(feat_func(game))
        return state_vect
    
    def opp_win(self, game):
        return winning_moves(self.opponent_color, game) #moves the op can make to win
    
    def my_win(self, game):
        return winning_moves(self.color, game)

    #moves that enables opp win
    def losing_moves(self,game: Game):
        board = game.board
        ret = []
        rows, cols = board.shape #get dimensions of board
        for c in range(cols):
            if game.is_valid_location(c):
                next_board = game.generate_successor(c, self.color)
                if next_board.is_valid_location(c) and next_board.generate_successor(c, self.opponent_color).winning_move(self.opponent_color):
                    ret.append(1)
                else:
                    ret.append(0)
            else:
                ret.append(0)
        return ret

    def opp_pieces_per_col(self, game):
        return pieces_per_col(self.opponent_color, game.board)

    def my_pieces_per_col(self, game):
        return pieces_per_col(self.color, game.board)

def pieces_per_col(color, board):
    ret = []
    rows, cols = board.shape #get dimensions of board
    loc = np.where(board == color, 1, 0)
    for c in range(cols):
        loc_T = np.transpose(loc)
        count = np.sum(loc_T[c]) / rows
        ret.append(count)
    return ret

def winning_moves(color, game : Game):
    board = game.board
    ret = []
    rows, cols = board.shape #get dimensions of board
    for c in range(cols):
        if game.is_valid_location(c) and game.generate_successor(c, color).winning_move(color):
            ret.append(1)
        else:
            ret.append(0)
    return ret

class AlphaBetaAgent(Agent):
    def __init__(self, color, depth) -> None:
        super().__init__(color)
        self.depth = depth

    def get_action(self, game):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        index = 0
        alpha = float("-inf")
        beta =  float("inf")
        result = self.alpha_beta(0, 0, alpha, beta, game)
        return result[1]
    # Credit to this post for giving us the idea of the AB evaluation function/check winning condition
    # https://stackoverflow.com/questions/29949169/python-connect-4-check-win-function
    def alpha_beta(self, depth, index, alpha, beta, game: Game):
        # terminal state, return evaluation function value
        legal_actions = game.get_valid_moves()
        if(len(legal_actions) == 0 or game.winning_move_faster(self.color) or game.winning_move_faster(self.opponent_color) or depth == self.depth * 2):
          return self.evaluation_function(game), None 
        v = 0
        action = None
        # Maximizing agent
        
        if index == 0:
            v = float("-inf")
            # iterate through legal actions for the agent
            # for alpha_beta, we only want to generate the states we need.
            for x in legal_actions:
                # generate (action, state) tuples
                pair = (x, game.generate_successor(x, self.color))
                oldVal = v
                # get the cost of the action for the next depth and next agent.
                costOfAction = self.alpha_beta(depth + 1, (index + 1) % 2, alpha, beta, pair[1])[0]
                v = max(v, costOfAction)
                if oldVal != v:
                    action = pair[0]
                if v > beta:
                    return v, action
                else:
                    alpha = max(alpha, v)
        # Minimizing agent
        else:
            v = float("inf")
            for x in legal_actions:
                pair = (x, game.generate_successor(x, self.opponent_color))
                oldVal = v
                costOfAction = self.alpha_beta(depth + 1, (index + 1) % 2, alpha, beta, pair[1])[0]
                v = min(v, costOfAction)
                if oldVal != v:
                    action = pair[0]
                if v < alpha:
                    return v, action
                else:
                    beta = min(beta, v)
        return v, action
    
    def evaluation_function(self, game: Game):
        score = 0.0

        horizontal_kernel = np.array([[1, 1, 1, 1]])
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(4, dtype=np.uint8)
        diag2_kernel = np.fliplr(diag1_kernel)
        kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]
        for kernel in kernels:
            if (convolve2d(game.board == self.color, kernel, mode="valid") == 4).any():
                score += 1000
            if (convolve2d(game.board == self.opponent_color, kernel, mode="valid") == 4).any():
                score -= 1000

        horizontal_kernel = np.array([[1, 1, 1]])
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(3, dtype=np.uint8)
        diag2_kernel = np.fliplr(diag1_kernel)
        kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]
        for kernel in kernels:
            if (convolve2d(game.board == self.color, kernel, mode="valid") == 3).any():
                score += 100
            if (convolve2d(game.board == self.opponent_color, kernel, mode="valid") == 3).any():
                score -= 100

        horizontal_kernel = np.array([[1, 1]])
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(2, dtype=np.uint8)
        diag2_kernel = np.fliplr(diag1_kernel)
        kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]
        for kernel in kernels:
            if (convolve2d(game.board == self.color, kernel, mode="valid") == 2).any():
                score += 10
            if (convolve2d(game.board == self.opponent_color, kernel, mode="valid") == 2).any():
                score -= 10

        return score
class RandomAlphaBeta(AlphaBetaAgent):
    def __init__(self, color, depth, depth_rand=0.5):
        super().__init__(color, depth)
        self.depth_rand = depth_rand
        
    
    def get_action(self, game):
        alpha = float("-inf")
        beta =  float("inf")
        change = 0
        if np.random.rand() < self.depth_rand:
            self.depth = self.depth + 1
            change += 1
        result = self.alpha_beta(0, 0, alpha, beta, game)
        self.depth = self.depth - change
        return result[1]
       
