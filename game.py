import numpy as np
import pygame
from scipy.signal import convolve2d 

BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)


class Game():
    def __init__(self, r, c, board=None):
        self.row_count = r
        self.col_count = c
        if board is None:
            self.board = np.zeros((r, c), dtype=np.int8)
        else:
            self.board = np.copy(board)



    def is_valid_location(self, col):
        return self.board[self.row_count-1][col] == 0

    def get_valid_moves(self):
        moves = []
        for a in range(self.col_count):
            if self.is_valid_location(a):
                moves.append(a)
        return moves
    def drop_piece(self, col, piece):
        row = self.get_next_open_row(col)
        self.board[row][col] = piece

    def get_next_open_row(self, col):
        for r in range(self.row_count):
            if self.board[r][col] == 0:
                return r
        return None
    def get_board_dimensions(self):
        return self.row_count, self.col_count

    def print_board(self):
        print(np.flip(self.board, 0))

    def generate_successor(self, col, piece):
        suc = Game(self.row_count, self.col_count, self.board)
        suc.drop_piece(col, piece)
        return suc
    # Credit to this post for giving us the idea of the AB evaluation function/check winning condition
    # https://stackoverflow.com/questions/29949169/python-connect-4-check-win-function
    def winning_move_faster(self, piece):
        horizontal_kernel = np.array([[1, 1, 1, 1]])
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(4, dtype=np.uint8)
        diag2_kernel = np.fliplr(diag1_kernel)
        kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]
        for kernel in kernels:
            if (convolve2d(self.board == piece, kernel, mode="valid") == 4).any():
                return True
        return False

    def on_perimeter_3_threat(self, color, coordinate):
        for c in range(self.col_count-3):
            for r in range(self.row_count):
                if self.board[r][c] == color and self.board[r][c+1] == color and self.board[r][c+2] == color and self.board[r][c+3] == color:
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



class FlipGame(Game):
    def __init__(self, r, c, board=None):
        super().__init__(r, c, board)
    
    def is_valid_location(self, col):
        return super().is_valid_location(col)

    def get_valid_moves(self):
        return super().get_valid_moves()

    def drop_piece(self, col, piece):
        return super().drop_piece(col, piece)

    def get_next_open_row(self, col):
        return super().get_next_open_row(col)
    
    def get_board_dimensions(self):
        return super().get_board_dimensions()
    
    def print_board(self):
        return super().print_board()
    
    def generate_successor(self, col, piece):
        return super().generate_successor(col, piece)
    
    def winning_move_faster(self, piece):
        return super().winning_move_faster(piece)
    
    def draw_board(self, screen, square_size):
        return super().draw_board(screen, square_size)