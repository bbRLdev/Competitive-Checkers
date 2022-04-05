import sys
from game import Game
import pygame
args = sys.argv
NUM_ROWS, NUM_COLS = int(args[1]), int(args[2])

game = Game(NUM_ROWS, NUM_COLS)

SQUARESIZE = 100
width = NUM_COLS * SQUARESIZE
height = (NUM_ROWS+1) * SQUARESIZE
size = (width, height)
screen = pygame.display.set_mode(size)

while True:
    game.draw_board(screen, SQUARESIZE)
