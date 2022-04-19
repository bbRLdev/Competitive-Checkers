import sys
from game import Game
import numpy as np
import pygame
from agent import Agent, RandomAgent, AlphaBetaAgent, SARSA_FeatureAgent
args = sys.argv
NUM_ROWS, NUM_COLS = int(args[1]), int(args[2])

game = Game(NUM_ROWS, NUM_COLS)

SQUARESIZE = 100
width = NUM_COLS * SQUARESIZE
height = (NUM_ROWS+1) * SQUARESIZE
size = (width, height)
screen = pygame.display.set_mode(size)

p1 = AlphaBetaAgent(1, 3)
p2 = RandomAgent(2)
for _ in range(10):
    game = Game(NUM_ROWS, NUM_COLS)
    while True:
        #p1 turn
        p1_move = p1.get_action(game)
        game.drop_piece(p1_move, 1)
        # game.draw_board(screen, SQUARESIZE)
        game.print_board()
        if game.winning_move(1):
            print("P1 Won!")
            break
        if len(game.get_valid_moves()) == 0:
            print("DRAW")
            break
        #p2 turn
        p2_move = p2.get_action(game)
        game.drop_piece(p2_move, 2)
        # game.draw_board(screen, SQUARESIZE)
        game.print_board()
        if game.winning_move(2):
            print("P2 Won!")
            break
        if len(game.get_valid_moves()) == 0:
            print("DRAW")
            break



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

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = game.col_count
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    w = np.zeros((X.feature_vector_len()))

    for _ in range(num_episode):
        observation = env.reset() #change
        done = False
        action = epsilon_greedy_policy(observation, done, w)
        x = X(observation, done, action)
        z = np.zeros(X.feature_vector_len())
        Q_old = 0
        while not done:
            observation, reward, done, _ = env.step(action) #chnage
            action_prime = epsilon_greedy_policy(observation, done, w)
            x_prime = X(observation, done, action_prime)
            # Q = np.transpose(w) * x
            # Q_prime = np.transpose(w) * x_prime
            # delta = reward + gamma * Q_prime - Q
            # z = lam * gamma * z + (1-(alpha * gamma * lam * np.transpose(z) * x)) * x
            Q = np.dot(w,x)
            Q_prime = np.dot(w, x_prime)
            delta = reward + gamma * Q_prime - Q
            z = lam * gamma * z + (1-(alpha * gamma * lam * np.dot(z,x))) * x
            w = w + alpha * (delta + Q - Q_old) * z - alpha * (Q - Q_old) * x
            Q_old = Q_prime
            x = x_prime
            action = action_prime

    return w