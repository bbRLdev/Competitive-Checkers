import sys
import numpy as np

args = sys.argv

NUM_ROWS = args[0]
NUM_COLS = args[1]
EMPTY = 0
BLUE = 1
RED = 2


board = np.zeros((NUM_ROWS, NUM_COLS), dtype=np.uint8)

