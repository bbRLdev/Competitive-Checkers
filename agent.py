from game import Game
import numpy as np
import math

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

class SARSA_FeatureAgent(Agent):
    def __init__(self, color) -> None:
        super().__init__(color)
        self.feature_functions = [self.opp_win(), self.my_win(), self.losing_moves(), self.opp_pieces_per_col(), self.my_pieces_per_col(), self.base3_rows(), self.base3_cols()]

    def get_action(self, game) -> int:
        vect = self.state_to_vect(game)

    def feature_vector_len(self):
        return 7 * len(self.feature_functions)

    def state_to_vect(self, game):
        state_vect = []
        for feat_func  in self.feature_functions:
            state_vect += feat_func(self, game)
    
    def opp_win(self, game):
        return winning_moves(self.opponent_color, game) #moves the op can make to win
    
    def my_win(self, game):
        return winning_moves(self.color, game)

    #moves that enables opp win
    def losing_moves(self,game):
        board = game.board
        ret = []
        rows, cols = board.shape() #get dimensions of board
        for c in range(cols):
            next_board = game.generate_successor(c, self.color)
            if next_board.generate_successor(c, self.opponent_color).winning_move(self.opponent_color):
                ret.append(1)
            else:
                ret.append(0)

    def opp_pieces_per_col(self, game):
        return pieces_per_col(self.opponent_color, game.board)

    def my_pieces_per_col(self, game):
        return pieces_per_col(self.color, game.board)

    def base3_rows(self, game):
        board = game.board
        ret = []
        rows, cols = board.shape() #get dimensions of board
        for r in range(rows):
            val = 0
            power = 0
            for c in range(cols):
                val += board[r][c] * math.pow(3, power)
                power += 1
            ret.append(val)
        return ret

    def base3_cols(self, game):
        board = game.board
        ret = []
        rows, cols = board.shape() #get dimensions of board
        for c in range(cols):
            val = 0
            power = 0
            for r in range(rows):
                val += board[r][c] * math.pow(3, power)
                power += 1
            ret.append(val)
        return ret

    # def base3_diags(self, board): #probs useless
    #     ret = []
    #     rows, cols = board.shape() #get dimensions of board

def pieces_per_col(color, board):
    ret = []
    print(board.shape())
    rows, cols = board.shape() #get dimensions of board
    loc = np.where(board == color, 1, 0)
    for c in range(cols):
        count = np.sum(loc[c])
        ret.append(count)
    return ret

def winning_moves(color, game : Game):
    board = game.board
    ret = []
    rows, cols = board.shape() #get dimensions of board
    for c in range(cols):
        if game.generate_successor(c, color).winning_move(color):
            ret.append(1)
        else:
            ret.append(0)


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
    
    def alpha_beta(self, depth, index, alpha, beta, game: Game):
        # terminal state, return evaluation function value
        if(game.winning_move(self.color) or game.winning_move(self.opponent_color) or depth == self.depth * 2):
          return self.evaluation_function(game), "" 
        v = 0
        action = ""
        # Maximizing agent
        if index == 0:
            v = float("-inf")
            # iterate through legal actions for the agent
            # for alpha_beta, we only want to generate the states we need.
            legalActions = game.get_valid_moves()
            for x in legalActions:
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
            legalActions = game.get_valid_moves()
            for x in legalActions:
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
        score = 0

        def check_window(window):
            score = 0
            player_count = 0
            opp_count = 0
            empty_count = 0
            for i in range(len(window)):
                if window[i] == self.color:
                    player_count += 1
                elif window[i] == 0:
                    empty_count += 1
                else:
                    opp_count +=1
            
            if player_count == 4:
                score = 100
            elif player_count == 3 and empty_count == 1:
                score = 5
            elif player_count == 2 and empty_count == 2:
                score = 2
            if opp_count == 3 and empty_count == 1:
                score -= 4
            return score
        
        #check horizontal
        for i in range(game.row_count):
            for j in range(game.col_count - 3):
                window = np.zeros(4)
                for k in range(j, j + 4):
                    window[k-j] = game.board[i][k]
                score += check_window(window)

        #score vertical
        for j in range(game.col_count):
            for i in range (game.row_count - 3):
                window = np.zeros(4)
                for k in range(i, i + 4):
                    window[k-i] = game.board[k][j]
                score += check_window(window)
        
        #score diagonals
        for i in range(game.row_count):
            for j in range(game.col_count):
                window = np.zeros(4)
                index = 0
                diagI = i
                diagJ = j
                while diagI >= 0 and diagJ < game.col_count and index < 4:
                    window[index] = game.board[diagI][diagJ]
                    diagI -= 1
                    diagJ += 1
                    index += 1
                if index == 4:
                    score += check_window(window)
                #other diag
                window = np.zeros(4)
                index = 0
                diagI = i
                diagJ = j
                while diagI < game.row_count and diagJ < game.col_count and index < 4:
                    window[index] = game.board[diagI][diagJ]
                    diagI += 1
                    diagJ += 1
                    index += 1
                if index == 4:
                    score += check_window(window)
            
            return score