import sys
from game import Game
import pygame
from agent import Agent, RandomAgent
args = sys.argv
NUM_ROWS, NUM_COLS = int(args[1]), int(args[2])

game = Game(NUM_ROWS, NUM_COLS)

SQUARESIZE = 100
width = NUM_COLS * SQUARESIZE
height = (NUM_ROWS+1) * SQUARESIZE
size = (width, height)
screen = pygame.display.set_mode(size)

p1 = RandomAgent()
p2 = RandomAgent()

while True:
    #p1 turn
    p1_move = p1.get_action(game)
    p1_row = game.get_next_open_row(p1_move)
    game.drop_piece(p1_row, p1_move, 1)
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
    p2_row = game.get_next_open_row(p2_move)
    game.drop_piece(p2_row, p2_move, 2)
    # game.draw_board(screen, SQUARESIZE)
    game.print_board()
    if game.winning_move(2):
        print("P2 Won!")
        break
    if len(game.get_valid_moves()) == 0:
        print("DRAW")
        break
