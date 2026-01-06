# HUMAN vs AI Tetris using the NEAT AI logic from your second script
# Human is on the left, AI on the right

import pygame
import random
import copy
from collections import deque
import numpy as np
import math
import pickle
import neat
import os

pygame.init()

CELL_SIZE = 30
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
MARGIN = 2
PREVIEW_WIDTH = 6
PREVIEW_HEIGHT = 4
SIDE_WIDTH = 150
SCREEN_WIDTH = int(2.5 * (BOARD_WIDTH * (CELL_SIZE + MARGIN) + SIDE_WIDTH))
SCREEN_HEIGHT = BOARD_HEIGHT * (CELL_SIZE + MARGIN)
AI_MOVES_PER_SECOND = 2
FPS = 60

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Tetris: Human vs AI")

BLACK = (0, 0, 0)
GREY = (50, 50, 50)
COLORS = [(0, 0, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255),
          (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 165, 0)]
GARBAGE_COLOR = (150, 150, 150)

TETROMINOES = {
    'I': [[[1, 1, 1, 1]], [[1], [1], [1], [1]]],
    'O': [[[1, 1], [1, 1]]],
    'T': [[[0,1,0],[1,1,1]],[[1,0],[1,1],[1,0]],[[1,1,1],[0,1,0]],[[0,1],[1,1],[0,1]]],
    'S': [[[0,1,1],[1,1,0]],[[1,0],[1,1],[0,1]]],
    'Z': [[[1,1,0],[0,1,1]],[[0,1],[1,1],[1,0]]],
    'J': [[[1,0,0],[1,1,1]],[[1,1],[1,0],[1,0]],[[1,1,1],[0,0,1]],[[0,1],[0,1],[1,1]]],
    'L': [[[0,0,1],[1,1,1]],[[1,0],[1,0],[1,1]],[[1,1,1],[1,0,0]],[[1,1],[0,1],[0,1]]]
}

TETROMINO_COLORS = {'I':1,'O':2,'T':3,'S':4,'Z':5,'J':6,'L':7}
PIECE_ORDER = list(TETROMINOES.keys())

class SevenBag:
    def __init__(self):
        self.bag = deque()
        self.fill_bag()
    def fill_bag(self):
        pieces = list(TETROMINOES.keys())
        random.shuffle(pieces)
        self.bag.extend(pieces)
    def next(self):
        if not self.bag:
            self.fill_bag()
        return self.bag.popleft()

class Board:
    def __init__(self):
        self.empty = [[0 for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        self.grid = [[0 for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
    def check_collision(self, shape, row, col):
        shape_h, shape_w = len(shape), len(shape[0])
        if col<0 or col+shape_w>BOARD_WIDTH or row+shape_h>BOARD_HEIGHT:
            return True
        for r in range(shape_h):
            for c in range(shape_w):
                if shape[r][c] and self.grid[row+r][col+c]:
                    return True
        return False
    def place(self, shape, row, col, color):
        for r in range(len(shape)):
            for c in range(len(shape[0])):
                if shape[r][c]:
                    self.grid[row+r][col+c] = color
    def clear_lines(self):
        cleared = 0
        new_grid = []
        for row in self.grid:
            if all(row):
                cleared += 1
            else:
                new_grid.append(row)
        while len(new_grid) < BOARD_HEIGHT:
            new_grid.insert(0, [0]*BOARD_WIDTH)
        self.grid = new_grid
        return cleared
    def add_garbage(self, n):
        for _ in range(n):
            self.grid.pop(0)
            hole = random.randint(0, BOARD_WIDTH-1)
            row = [8]*BOARD_WIDTH
            row[hole] = 0
            self.grid.append(row)
    def draw(self, offset_x):
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                val = self.grid[r][c]
                color = GARBAGE_COLOR if val==8 else COLORS[val]
                pygame.draw.rect(screen,color,(offset_x+c*(CELL_SIZE+MARGIN),
                                              r*(CELL_SIZE+MARGIN),
                                              CELL_SIZE,CELL_SIZE))
    def as_numpy(self):
        arr = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                arr[r,c] = self.grid[r][c]
        return arr

class Player:
    def __init__(self, offset_x):
        self.board = Board()
        self.bag = SevenBag()
        self.current_piece = self.new_piece()
        self.row = 0
        self.col = BOARD_WIDTH//2 - len(self.current_piece[0])//2
        self.offset_x = offset_x
        self.hold_piece = None
        self.can_hold = True
        self.next_piece_preview = self.new_piece()
        self.garbage_queue = 0
        self.back_to_back = False
        self.combo = 0
        self.game_over = False
        self.DAS_DELAY = 10
        self.ARR_RATE = 2
    def new_piece(self):
        piece_type = self.bag.next()
        shape = copy.deepcopy(TETROMINOES[piece_type][0])
        color = TETROMINO_COLORS[piece_type]
        return (shape,color,piece_type)
    def rotate(self):
        shape,color,name = self.current_piece
        rotated = [list(row) for row in zip(*shape[::-1])]
        if not self.board.check_collision(rotated,self.row,self.col):
            self.current_piece = (rotated,color,name)
    def move(self,dx):
        shape,color,_ = self.current_piece
        if not self.board.check_collision(shape,self.row,self.col+dx):
            self.col += dx
    def soft_drop(self):
        shape,color,_ = self.current_piece
        if not self.board.check_collision(shape,self.row+1,self.col):
            self.row +=1
            return 0
        else:
            return self.lock_piece()
    def hard_drop(self):
        shape,color,_ = self.current_piece
        while not self.board.check_collision(shape,self.row+1,self.col):
            self.row +=1
        return self.lock_piece()
    def hold(self):
        if not self.can_hold:
            return
        self.can_hold = False
        if self.hold_piece is None:
            self.hold_piece = self.current_piece[2]
            self.current_piece = self.next_piece_preview
            self.next_piece_preview = self.new_piece()
        else:
            temp = self.hold_piece
            self.hold_piece = self.current_piece[2]
            shape = copy.deepcopy(TETROMINOES[temp][0])
            color = TETROMINO_COLORS[temp]
            self.current_piece = (shape,color,temp)
        self.row = 0
        self.col = BOARD_WIDTH//2 - len(self.current_piece[0])//2
    def lock_piece(self):
        shape,color,_ = self.current_piece
        if self.board.check_collision(shape,self.row,self.col) and self.row==0:
            self.game_over=True
            return 0
        self.board.place(shape,self.row,self.col,color)
        lines = self.board.clear_lines()
        garbage = self.calculate_garbage(lines)
        self.current_piece = self.next_piece_preview
        self.next_piece_preview = self.new_piece()
        self.row=0
        self.col=BOARD_WIDTH//2 - len(self.current_piece[0])//2
        self.can_hold=True
        return garbage
    def calculate_garbage(self,lines):
        g=0
        if lines==0:
            self.combo=0
        elif lines==1:
            self.combo += 0.5
            self.back_to_back=False
        elif lines==2:
            g+=1
            self.combo+=1
            self.back_to_back=False
        elif lines==3:
            g+=2
            self.combo+=1
            self.back_to_back=False
        elif lines==4:
            g+=4
            if self.back_to_back:
                g+=1
            self.back_to_back=True
            self.combo+=1
        if self.combo>1:
            g+=(self.combo//2)
        if (self.board.grid == self.board.empty):
            g += 10
        return int(g)
    def apply_garbage(self):
        if self.garbage_queue>0:
            self.board.add_garbage(self.garbage_queue)
            self.garbage_queue=0
    def draw_preview(self,piece,offset_x,offset_y):
        shape,color,_ = piece
        for r in range(len(shape)):
            for c in range(len(shape[0])):
                if shape[r][c]:
                    pygame.draw.rect(screen,COLORS[color],(offset_x+c*(CELL_SIZE+MARGIN),
                                                           offset_y+r*(CELL_SIZE+MARGIN),
                                                           CELL_SIZE,CELL_SIZE))
    def draw_garbage(self,offset_x,offset_y,rect):
        font = pygame.font.Font(None,40)
        surf = font.render(str(self.garbage_queue),True,(255,255,255))
        pygame.draw.rect(screen,(0,0,0),rect.inflate(20,20))
        screen.blit(surf,rect)
    def draw(self):
        left_edge = self.offset_x+SIDE_WIDTH
        right_edge = left_edge + BOARD_WIDTH*(CELL_SIZE+MARGIN)
        pygame.draw.line(screen,GREY,(left_edge-2,0),(left_edge-2,SCREEN_HEIGHT),2)
        pygame.draw.line(screen,GREY,(right_edge+2,0),(right_edge+2,SCREEN_HEIGHT),2)
        self.board.draw(left_edge)
        shape,color,_ = self.current_piece
        for r in range(len(shape)):
            for c in range(len(shape[0])):
                if shape[r][c]:
                    pygame.draw.rect(screen,COLORS[color],
                                     (left_edge+(self.col+c)*(CELL_SIZE+MARGIN),
                                      (self.row+r)*(CELL_SIZE+MARGIN),
                                      CELL_SIZE,CELL_SIZE))
        is_player2 = self.offset_x>SCREEN_WIDTH//4
        hold_x = right_edge+20 if is_player2 else self.offset_x-40
        next_x = hold_x
        garbage_x = right_edge+30 if is_player2 else self.offset_x-50
        pygame.draw.rect(screen,GREY,(hold_x,10,PREVIEW_WIDTH*CELL_SIZE,PREVIEW_HEIGHT*CELL_SIZE),2)
        if self.hold_piece:
            shape = copy.deepcopy(TETROMINOES[self.hold_piece][0])
            color = TETROMINO_COLORS[self.hold_piece]
            self.draw_preview((shape,color,self.hold_piece),hold_x,10)
        pygame.draw.rect(screen,GREY,(next_x,150,PREVIEW_WIDTH*CELL_SIZE,PREVIEW_HEIGHT*CELL_SIZE),2)
        self.draw_preview(self.next_piece_preview,next_x,150)
        rect = pygame.draw.rect(screen,GREY,(garbage_x,290,20,20),2)
        self.draw_garbage(garbage_x,290,rect)

class AIPlayer(Player):
    def __init__(self, offset_x, net=None):
        super().__init__(offset_x)
        self.net = net
        self.hold_locked = False
        self.just_held = False

    def ai_take_action_and_drop(self):
        if self.net is None:
            print("NET NOT FOUND!")
            piece_name = self.current_piece[2]
            rotations = TETROMINOES[piece_name]
            for rot in rotations:
                for col in range(BOARD_WIDTH - len(rot[0]) + 1):
                    if not self.board.check_collision(rot, 0, col):
                        self.current_piece = (rot, TETROMINO_COLORS[piece_name], piece_name)
                        self.row = 0
                        self.col = col
                        return self.hard_drop()
            return 0
        
        if self.just_held:
            self.just_held = False
            return 0

        board_np = create_board_numpy_from_grid(self.board.grid)
        best_move = get_best_move(
            board_np,
            self.current_piece[2],
            self.next_piece_preview[2],
            self.hold_piece,
            self.hold_locked,
            self.garbage_queue,
            self.net
        )


        if best_move is None:
            print("NO LEGAL MOVES FOUND!")
            return self.lock_piece()

        if best_move['action_type'] in ['hold_next', 'hold_swap']:
            self.hold()
            self.hold_locked = True
            self.just_held = True
            return 0

        else:
            self.hold_locked = False

        self.current_piece = (best_move['rotation'], TETROMINO_COLORS[best_move['piece']], best_move['piece'])
        self.row = 0
        self.col = best_move['col']
        return self.hard_drop()

def one_hot_piece(piece):
    vec = [0]*len(PIECE_ORDER)
    if piece is None:
        return vec
    idx = PIECE_ORDER.index(piece)
    vec[idx] = 1
    return vec

def get_features(b, cur, nxt, held, hold_locked, garbage_q):
    heights = [next((BOARD_HEIGHT-r for r in range(BOARD_HEIGHT) if b[r,c]),0)
               for c in range(BOARD_WIDTH)]
    holes = sum(b[r,c]==0 and any(b[:r,c])
                for c in range(BOARD_WIDTH) for r in range(BOARD_HEIGHT))
    bump = sum(abs(heights[i]-heights[i+1]) for i in range(9))
    avg_h = sum(heights)/10

    return (
        heights +
        [holes, bump, avg_h] +
        one_hot_piece(cur) +
        one_hot_piece(nxt) +
        one_hot_piece(held) +
        [1 if hold_locked else 0] +
        [garbage_q]
    )


def get_best_move(board_numpy, current_piece, next_piece, held_piece, hold_locked, garbage_q, net):
    """
    Returns the best move with hold logic as a dict:
        {'action_type': ..., 'piece': ..., 'rotation': ..., 'col': ...}
    """
    best_score = -math.inf
    best_action = None

    def evaluate_piece(piece, action_type):
        nonlocal best_score, best_action
        for rotation in TETROMINOES[piece]:
            shape = np.array(rotation)
            for col in range(board_numpy.shape[1] - shape.shape[1] + 1):
                if check_collision_np(board_numpy, shape, 0, col):
                    continue
                piece_val = TETROMINO_COLORS[piece]
                lines_cleared, new_board = place_piece_np(board_numpy.copy(), shape, col, piece_val)
                if lines_cleared == -1:
                    continue
                features = get_features(
                    new_board,
                    piece,
                    next_piece,
                    held_piece,
                    action_type != "current",
                    garbage_q
                )

                try:
                    score = net.activate([float(f) for f in features])[0]
                except Exception:
                    score = -sum(features)
                if score > best_score:
                    best_score = score
                    best_action = {
                        'action_type': action_type,
                        'piece': piece,
                        'rotation': [list(row) for row in rotation],
                        'col': col
                    }

    # Current piece
    evaluate_piece(current_piece, 'current')

    # Hold -> use next piece
    if not hold_locked and next_piece:
        evaluate_piece(next_piece, 'hold_next')

    # Hold -> use held piece
    if not hold_locked and held_piece:
        evaluate_piece(held_piece, 'hold_swap')

    return best_action


def check_collision_np(board, shape, row, col):
    """
    Returns True if the piece shape collides at (row, col) on board.
    """
    shape = np.array(shape)
    if row + shape.shape[0] > board.shape[0] or col < 0 or col + shape.shape[1] > board.shape[1]:
        return True
    return np.any((shape == 1) & (board[row:row+shape.shape[0], col:col+shape.shape[1]] != 0))

def clear_lines_np(board):
    non_full = board[np.any(board == 0, axis=1)]
    cleared = BOARD_HEIGHT - non_full.shape[0]

    if cleared > 0:
        board[:] = np.vstack((
            np.zeros((cleared, BOARD_WIDTH), dtype=int),
            non_full
        ))

    return cleared


def create_board_numpy_from_grid(grid):
    """
    Converts a Board.grid (list of lists) into a numpy array.
    """
    arr = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
    for r in range(BOARD_HEIGHT):
        for c in range(BOARD_WIDTH):
            arr[r, c] = grid[r][c]
    return arr

def place_piece_np(board, shape, col, piece_val):
    """
    Places a piece on the board at the given column and returns (lines_cleared, new_board).
    piece_val must be an integer (color), not the piece name.
    """
    shape = np.array(shape)
    row = 0
    for r in range(board.shape[0]):
        if check_collision_np(board, shape, r, col):
            row = r - 1
            break
    else:
        row = board.shape[0] - shape.shape[0]

    if row < 0:
        return -1, board

    newb = board.copy()
    for r in range(shape.shape[0]):
        for c in range(shape.shape[1]):
            if shape[r, c]:
                newb[row + r, col + c] = piece_val

    lines_cleared = clear_lines_np(newb)
    return lines_cleared, newb

def load_ai_network(genome_path=os.path.join(os.path.dirname(__file__), "best_tetris_ai.pkl"), config_path=os.path.join(os.path.dirname(__file__), "neat-config.txt")):
    try:
        with open(genome_path, "rb") as f:
            genome = pickle.load(f)
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        print("AI network loaded.")
        return net, config
    except Exception as e:
        print("Failed to load AI network:", e)
        return None, None

def draw_winner(text):
    font = pygame.font.Font(None, 80)
    surf = font.render(text, True, (255, 255, 255))
    rect = surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
    pygame.draw.rect(screen, (0, 0, 0), rect.inflate(40, 20))
    screen.blit(surf, rect)
    pygame.display.flip()


def handle_das(player, left_key, right_key, keys, prev_keys):
    if keys[left_key] and not keys[right_key]:
        dir_pressed = -1
    elif keys[right_key] and not keys[left_key]:
        dir_pressed = 1
    else:
        dir_pressed = 0

    if dir_pressed != 0:
        if dir_pressed != player.das_dir:
            player.das_dir = dir_pressed
            player.das_timer = player.DAS_DELAY
            player.arr_timer = 0
            player.move(dir_pressed)
        else:
            if player.das_timer > 0:
                player.das_timer -= 1
            else:
                player.arr_timer -= 1
                if player.arr_timer <= 0:
                    player.move(dir_pressed)
                    player.arr_timer = player.ARR_RATE
    else:
        player.das_dir = 0
        player.das_timer = 0
        player.arr_timer = 0

# ---------- Main game loop ----------
def main():
    clock = pygame.time.Clock()
    game = 0

    while game < 5:
        net, config = load_ai_network()
        player1 = Player(90)
        player2 = AIPlayer(450, net=net)  # AI on the right

        running = True
        move = 0
        last_garbage_change = 0

        prev_keys = pygame.key.get_pressed()

        while running:
            screen.fill(BLACK)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()

            # --- Player 1 (human) ---
            if not player1.game_over:
                garbage = 0
                handle_das(player1, pygame.K_LEFT, pygame.K_RIGHT, keys, prev_keys)
                if keys[pygame.K_UP] and not prev_keys[pygame.K_UP]: player1.rotate()
                if keys[pygame.K_DOWN] and move%3 == 0: garbage += player1.soft_drop()
                if keys[pygame.K_SLASH] and not prev_keys[pygame.K_SLASH]: player1.hold()
                if keys[pygame.K_RSHIFT] and not prev_keys[pygame.K_RSHIFT]: garbage += player1.hard_drop()

                # gravity
                if (move%30 == 0):
                    shape, color, _ = player1.current_piece
                    if not player1.board.check_collision(shape, player1.row+1, player1.col):
                        player1.row += 1
                    else:
                        garbage += player1.lock_piece()
                player2.garbage_queue += garbage
                if (garbage > 0):
                    last_garbage_change = move
            else:
                pygame.time.wait(1500)
                draw_winner("Player 2 Wins!")
                pygame.time.wait(3000)
                running = False

            # --- Player 2 (AI) ---
            if not player2.game_over:
                garbage = 0
                if move % (FPS/AI_MOVES_PER_SECOND) == 0 and player1.garbage_queue < 25:
                    garbage += player2.ai_take_action_and_drop()
                else:
                    if (move%30 == 0):
                        shape, color, _ = player2.current_piece
                        if not player2.board.check_collision(shape, player2.row+1, player2.col):
                            player2.row += 1
                        else:
                            garbage += player2.lock_piece()

                player1.garbage_queue += garbage
                if ((player1.garbage_queue != 0 or player2.garbage_queue != 0) and not garbage_zerochanged):
                    last_garbage_nonzero = move
                    garbage_zerochanged = True
            else:
                pygame.time.wait(1500)
                draw_winner("Player 1 Wins!")
                pygame.time.wait(3000)
                running = False

            tot_garb = abs(player1.garbage_queue - player2.garbage_queue)
            if (player1.garbage_queue > player2.garbage_queue):
                player1.garbage_queue = tot_garb
                player2.garbage_queue = 0
            elif (player2.garbage_queue > player1.garbage_queue):
                player2.garbage_queue = tot_garb
                player1.garbage_queue = 0

            if (move - last_garbage_change > FPS*4):
                player1.apply_garbage()
                player2.apply_garbage()

            player1.draw()
            player2.draw()

            pygame.display.flip()
            clock.tick(FPS)
            move += 1
            prev_keys = keys

        game += 1
    pygame.quit()

if __name__ == "__main__":
    main()
