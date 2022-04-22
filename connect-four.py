import sys
from game import Game
import numpy as np
import pygame
from agent import Agent, LegacyAlphaBeta, RandomAgent, AlphaBetaAgent, SARSA_FeatureAgent
from tqdm import tqdm
args = sys.argv
NUM_ROWS, NUM_COLS = int(args[1]), int(args[2])

def SarsaLambda(
    game: Game, # connect-4 game
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:SARSA_FeatureAgent,
    Opponent:Agent,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(game,done,w,epsilon=.0):
        nA = game.col_count
        Q = [np.dot(w, X(game,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            valid_moves = game.get_valid_moves()
            index = np.random(len(valid_moves))
            return valid_moves[index]
        else:
            act = np.argmax(Q)
            while not game.is_valid_location(act):
                Q[act] = np.NINF
                act = np.argmax(Q)
            return act

    w = np.zeros((X.feature_vector_len() * game.col_count), dtype=np.float32)
    win_count = 0
    boards = []
    for _ in tqdm(range(num_episode)):
        observation = Game(NUM_ROWS, NUM_COLS) #change game reset
        boards.append(observation.board)
        done = len(game.get_valid_moves()) == 0
        action = epsilon_greedy_policy(observation, done, w)
        x = X(observation, done, action)
        z = np.zeros(X.feature_vector_len() * observation.col_count)
        Q_old = 0
        while not done:
            observation, reward, done, win_count = step(observation, action, Opponent, win_count) #chnage
            boards.append(observation.board)
            action_prime = epsilon_greedy_policy(observation, done, w)
            x_prime = X(observation, done, action_prime)
            Q = np.dot(w,x)
            Q_prime = np.dot(w, x_prime)
            delta = reward + gamma * Q_prime - Q
            z = lam * gamma * z + (1-(alpha * gamma * lam * np.dot(z,x))) * x
            w = w + alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x
            Q_old = Q_prime
            x = x_prime
            action = action_prime

    # X.set_weights(w)
    print("win count: ", win_count)
    # print(boards)
    return w

def step(game : Game, action, opponent : Agent, win_count):
    player_color = opponent.opponent_color
    game.drop_piece(action, player_color)
    # game.print_board()
    reward = 0.0
    if game.winning_move_faster(player_color):
        reward = 1.0
        win_count += 1
        # game.print_board()
        return game, reward, True, win_count
    opponent_action = opponent.get_action(game)
    #tie
    if opponent_action == None:
        return game, 0.0, True, win_count
    game.drop_piece(opponent_action, opponent.color)
    # game.print_board()
    if game.winning_move_faster(opponent.color):
        reward = -1.0
        return game, reward, True, win_count
    return game, 0.0, len(game.get_valid_moves()) == 0, win_count


game = Game(NUM_ROWS, NUM_COLS)

# SQUARESIZE = 100
# width = NUM_COLS * SQUARESIZE
# height = (NUM_ROWS+1) * SQUARESIZE
# size = (width, height)
# screen = pygame.display.set_mode(size)

alpha = AlphaBetaAgent(2, 2)
random = RandomAgent(2)
sarsa_agent = SARSA_FeatureAgent(1, NUM_COLS)
print(SarsaLambda(
    game, # connect-4 game
    0.8, # discount factor
    0.1, # decay rate
    0.001, # step size
    sarsa_agent,
    alpha, #opponent
    1000, # episode
)) 

# p1 = AlphaBetaAgent(1, 2)
# p2 = LegacyAlphaBeta(2, 2)
# for _ in range(10):
#     game = Game(NUM_ROWS, NUM_COLS)
#     while True:
#         #p1 turn
#         p1_move = p1.get_action(game)
#         game.drop_piece(p1_move, 1)
#         # game.draw_board(screen, SQUARESIZE)
#         game.print_board()
#         if game.winning_move_faster(1):
#             print("P1 Won!")
#             break
#         if len(game.get_valid_moves()) == 0:
#             print("DRAW")
#             break
#         #p2 turn
#         p2_move = p2.get_action(game)
#         game.drop_piece(p2_move, 2)
#         # game.draw_board(screen, SQUARESIZE)
#         game.print_board()
#         if game.winning_move_faster(2):
#             print("P2 Won!")
#             break
#         if len(game.get_valid_moves()) == 0:
#             print("DRAW")
#             break
