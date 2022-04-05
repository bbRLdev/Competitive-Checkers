import numpy as np
import pygame


BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)

class Game():
    def __init__(self, r, c):
        self.row_count = r
        self.col_count = c
        self.board = np.zeros((r, c), dtype=np.int8)

    def drop_piece(self, row, col, piece):
        self.board[row][col] = piece

    def is_valid_location(self, col):
        return self.board[self.row_count-1][col] == 0

    def get_next_open_row(self, col):
        for r in range(self.row_count):
            if self.board[r][col] == 0:
                return r

    def print_board(self):
        print(np.flip(self.board, 0))

    def winning_move(self, piece):
        # Check horizontal locations for win
        for c in range(self.col_count-3):
            for r in range(self.row_count):
                if self.board[r][c] == piece and self.board[r][c+1] == piece and self.board[r][c+2] == piece and self.board[r][c+3] == piece:
                    return True

        # Check vertical locations for win
        for c in range(self.col_count):
            for r in range(self.row_count-3):
                if self.board[r][c] == piece and self.board[r+1][c] == piece and self.board[r+2][c] == piece and self.board[r+3][c] == piece:
                    return True

        # Check positively sloped diaganols
        for c in range(self.col_count-3):
            for r in range(self.row_count-3):
                if self.board[r][c] == piece and self.board[r+1][c+1] == piece and self.board[r+2][c+2] == piece and self.board[r+3][c+3] == piece:
                    return True

        # Check negatively sloped diaganols
        for c in range(self.col_count-3):
            for r in range(3, self.row_count):
                if self.board[r][c] == piece and self.board[r-1][c+1] == piece and self.board[r-2][c+2] == piece and self.board[r-3][c+3] == piece:
                    return True

    def draw_board(self, screen, square_size):
        radius = int(square_size/2 - 5)
        height = (self.row_count+1) * square_size
        for c in range(self.col_count):
            for r in range(self.row_count):
                pygame.draw.rect(screen, BLUE, (c*square_size, r*square_size+square_size, square_size, square_size))
                pygame.draw.circle(screen, BLACK, (int(c*square_size+square_size/2), int(r*square_size+square_size+square_size/2)), radius)
        
        for c in range(self.col_count):
            for r in range(self.row_count):		
                if self.board[r][c] == 1:
                    pygame.draw.circle(screen, RED, (int(c*square_size+square_size/2), height-int(r*square_size+square_size/2)), radius)
                elif self.board[r][c] == 2: 
                    pygame.draw.circle(screen, YELLOW, (int(c*square_size+square_size/2), height-int(r*square_size+square_size/2)), radius)
        pygame.display.update()



