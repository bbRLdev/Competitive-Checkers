import numpy as np
import sys
from tqdm import tqdm
from agent import AlphaBetaAgent, SARSA_FeatureAgent, RandomAgent, LegacyAlphaBeta, HumanAgent
from game import Game

args = sys.argv
NUM_ROWS, NUM_COLS = int(args[1]), int(args[2])

w = np.load("out.npy")

print(w)

p1 = SARSA_FeatureAgent(1, 7)
p1.set_weights(w)

p2 = HumanAgent(2)
wins = 0
# p2 = RandomAgent(2)
for _ in tqdm(range(10)):
    game = Game(NUM_ROWS, NUM_COLS)
    while True:
        #p1 turn
        p1_move = p1.get_action(game)
        game.drop_piece(p1_move, 1)
        # game.draw_board(screen, SQUARESIZE)
        game.print_board()
        if game.winning_move_faster(1):
            print("P1 Won!")
            wins += 1
            break
        if len(game.get_valid_moves()) == 0:
            print("DRAW")
            break
        #p2 turn
        p2_move = p2.get_action(game)
        game.drop_piece(p2_move, 2)
        # game.draw_board(screen, SQUARESIZE)
        game.print_board()
        if game.winning_move_faster(2):
            print("P2 Won!")
            break
        if len(game.get_valid_moves()) == 0:
            print("DRAW")
            break

print(wins)