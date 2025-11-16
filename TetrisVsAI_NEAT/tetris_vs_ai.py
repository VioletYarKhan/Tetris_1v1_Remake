import pygame
import random
import copy
from collections import deque, Counter
import numpy as np
import math
import pickle
import os

try:
    import neat
    HAVE_NEAT = True
except Exception:
    HAVE_NEAT = False

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
AI_MOVES_PER_SECOND = 0.5
FPS = 60

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("2-Player Tetris vs AI")

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

TETROMINO_COLORS = {
    'I': 1, 'O': 2, 'T':3, 'S':4, 'Z':5, 'J':6, 'L':7
}

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
        self.grid = [[0 for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]

    def check_collision(self, shape, row, col):
        shape_h = len(shape)
        shape_w = len(shape[0])
        if col < 0 or col + shape_w > BOARD_WIDTH or row + shape_h > BOARD_HEIGHT:
            return True
        for r in range(shape_h):
            for c in range(shape_w):
                if shape[r][c] and self.grid[row + r][col + c]:
                    return True
        return False

    def place(self, shape, row, col, color):
        for r in range(len(shape)):
            for c in range(len(shape[0])):
                if shape[r][c]:
                    self.grid[row + r][col + c] = color

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
                if val == 8:
                    color = GARBAGE_COLOR
                else:
                    color = COLORS[val]
                pygame.draw.rect(screen, color, (offset_x + c*(CELL_SIZE+MARGIN),
                                                 r*(CELL_SIZE+MARGIN),
                                                 CELL_SIZE,
                                                 CELL_SIZE))

    def as_numpy(self):
        arr = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=object)
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                arr[r, c] = self.grid[r][c]
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
        self.das_dir = 0
        self.das_timer = 0
        self.arr_timer = 0
        self.DAS_DELAY = 10
        self.ARR_RATE = 2

    def new_piece(self):
        piece_type = self.bag.next()
        shape = copy.deepcopy(TETROMINOES[piece_type][0])
        color = TETROMINO_COLORS[piece_type]
        return (shape, color, piece_type)

    def rotate(self):
        shape, color, name = self.current_piece
        rotated = list(zip(*shape[::-1]))
        rotated = [list(row) for row in rotated]
        if not self.board.check_collision(rotated, self.row, self.col):
            self.current_piece = (rotated, color, name)

    def move(self, dx):
        shape, color, _ = self.current_piece
        if not self.board.check_collision(shape, self.row, self.col + dx):
            self.col += dx

    def soft_drop(self):
        shape, color, _ = self.current_piece
        if not self.board.check_collision(shape, self.row+1, self.col):
            self.row += 1
            return 0
        else:
            return self.lock_piece()

    def hard_drop(self):
        shape, color, _ = self.current_piece
        while not self.board.check_collision(shape, self.row+1, self.col):
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
            self.current_piece = (shape, color, temp)
        self.row = 0
        self.col = BOARD_WIDTH//2 - len(self.current_piece[0])//2

    def lock_piece(self):
        shape, color, _ = self.current_piece
        if self.board.check_collision(shape, self.row, self.col) and self.row == 0:
            self.game_over = True
            return 0
        self.board.place(shape, self.row, self.col, color)
        lines_cleared = self.board.clear_lines()
        garbage_sent = self.calculate_garbage(lines_cleared)
        self.current_piece = self.next_piece_preview
        self.next_piece_preview = self.new_piece()
        self.row = 0
        self.col = BOARD_WIDTH//2 - len(self.current_piece[0])//2
        self.can_hold = True
        return garbage_sent

    # Majorly Simplified from actual modern tetris
    def calculate_garbage(self, lines_cleared):
        garbage = 0
        if lines_cleared == 0:
            self.combo = 0
        elif lines_cleared == 1:
            self.combo += 1
            if self.combo > 2:
                garbage += 1
            self.back_to_back = False
        elif lines_cleared == 2:
            garbage += 1
            self.combo += 1
            self.back_to_back = False
        elif lines_cleared == 3:
            garbage += 2
            self.combo += 1
            self.back_to_back = False
        elif lines_cleared == 4:
            garbage += 4
            if self.back_to_back:
                garbage += 1
            self.back_to_back = True
            self.combo += 1
        if self.combo > 1:
            garbage += self.combo 
        return garbage

    def apply_garbage(self):
        if self.garbage_queue > 0:
            self.board.add_garbage(self.garbage_queue)
            self.garbage_queue = 0

    def draw_preview(self, piece, offset_x, offset_y):
        shape, color, _ = piece
        for r in range(len(shape)):
            for c in range(len(shape[0])):
                if shape[r][c]:
                    pygame.draw.rect(screen, COLORS[color],
                                     (offset_x + c*(CELL_SIZE+MARGIN),
                                      offset_y + r*(CELL_SIZE+MARGIN),
                                      CELL_SIZE, CELL_SIZE))
                    
    def draw_garbage(self, offset_x, offset_y, rect):
        font = pygame.font.Font(None, 40)
        surf = font.render(str(self.garbage_queue), True, (255, 255, 255))
        pygame.draw.rect(screen, (0, 0, 0), rect.inflate(20, 20))
        screen.blit(surf, rect)

    def draw(self):
        # Board background and frame
        left_edge = self.offset_x + SIDE_WIDTH
        right_edge = left_edge + BOARD_WIDTH * (CELL_SIZE + MARGIN)

        # Draw left divider
        pygame.draw.line(screen, GREY, (left_edge - 2, 0),
                        (left_edge - 2, SCREEN_HEIGHT), 2)
        # Draw right divider
        pygame.draw.line(screen, GREY, (right_edge + 2, 0),
                        (right_edge + 2, SCREEN_HEIGHT), 2)

        # Draw board
        self.board.draw(left_edge)

        # Current piece
        shape, color, _ = self.current_piece
        for r in range(len(shape)):
            for c in range(len(shape[0])):
                if shape[r][c]:
                    pygame.draw.rect(screen, COLORS[color],
                                    (left_edge + (self.col + c) * (CELL_SIZE + MARGIN),
                                    (self.row + r) * (CELL_SIZE + MARGIN),
                                    CELL_SIZE, CELL_SIZE))

        is_player2 = self.offset_x > SCREEN_WIDTH // 4

        if is_player2:
            hold_x = right_edge + 20
            next_x = right_edge + 20
            garbage_x = right_edge + 30
        else:
            hold_x = self.offset_x - 40
            next_x = self.offset_x - 40
            garbage_x = self.offset_x - 50

        # Hold
        pygame.draw.rect(screen, GREY, (hold_x, 10,
                                        PREVIEW_WIDTH * CELL_SIZE,
                                        PREVIEW_HEIGHT * CELL_SIZE), 2)
        if self.hold_piece:
            shape = copy.deepcopy(TETROMINOES[self.hold_piece][0])
            color = TETROMINO_COLORS[self.hold_piece]
            self.draw_preview((shape, color, self.hold_piece), hold_x, 10)

        # Next
        pygame.draw.rect(screen, GREY, (next_x, 150,
                                        PREVIEW_WIDTH * CELL_SIZE,
                                        PREVIEW_HEIGHT * CELL_SIZE), 2)
        self.draw_preview(self.next_piece_preview, next_x, 150)

        # Garbage
        rect = pygame.draw.rect(screen, GREY, (garbage_x, 290,
                                        20,
                                        20), 2)
        self.draw_garbage(garbage_x, 290, rect)

def create_board_numpy_from_grid(grid):
    arr = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=object)
    for r in range(BOARD_HEIGHT):
        for c in range(BOARD_WIDTH):
            arr[r, c] = grid[r][c]
    return arr

def check_collision_np(board, shape, row, col):
    shape = np.array(shape)
    if row + shape.shape[0] > BOARD_HEIGHT or col < 0 or col + shape.shape[1] > BOARD_WIDTH:
        return True
    return np.any((shape == 1) & (board[row:row + shape.shape[0], col:col + shape.shape[1]] != 0))

def place_piece_np(board, shape, col, piece_val):
    shape = np.array(shape)
    row = 0
    for r in range(BOARD_HEIGHT):
        if check_collision_np(board, shape, r, col):
            row = r - 1
            break
    else:
        row = BOARD_HEIGHT - shape.shape[0]
    if row < 0:
        return -1, -1
    newb = board.copy()
    for r in range(shape.shape[0]):
        for c in range(shape.shape[1]):
            if shape[r][c]:
                newb[row + r, col + c] = piece_val
    lines = clear_lines_np(newb)
    return lines, newb

def clear_lines_np(board):
    full_rows = np.where(np.all(board != 0, axis=1))[0]
    if full_rows.size == 0:
        return 0
    for r in full_rows:
        if r > 0:
            board[1:r+1] = board[0:r]
        board[0] = 0
    return full_rows.size

def get_features(board):
    heights = []
    for c in range(BOARD_WIDTH):
        col_height = 0
        for r in range(BOARD_HEIGHT):
            if board[r, c] != 0:
                col_height = BOARD_HEIGHT - r
                break
        heights.append(col_height)
    holes = 0
    for c in range(BOARD_WIDTH):
        block_seen = False
        for r in range(BOARD_HEIGHT):
            if board[r, c] != 0:
                block_seen = True
            elif block_seen and board[r, c] == 0:
                holes += 1
    bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))
    avg_height = sum(heights)/len(heights)
    return heights + [holes, bumpiness, avg_height]

def get_best_move_with_hold(board_np, current_piece, next_piece, hold_piece, net):
    """
    Returns a tuple:
        (best_score, action_type, piece_name, rotation, col)

    action_type is one of:
        "current"   -> place current_piece
        "hold_next" -> hold, then use next_piece
        "hold_swap" -> hold swap, use hold_piece
    """

    best_score = -math.inf
    best = ("current", current_piece, None, None)  # action_type, piece, rot, col

    # Try current piece
    action = get_best_move_for_piece(board_np, current_piece, net)
    if action:
        rot, col = action
        shape = [list(row) for row in rot]
        lines, newb = place_piece_np(board_np.copy(), shape, col, current_piece)
        feats = get_features(newb)
        score = net.activate([float(x) for x in feats])[0]
        if score > best_score:
            best_score = score
            best = ("current", current_piece, rot, col)

    # Try holding -> use next
    if next_piece:
        action = get_best_move_for_piece(board_np, next_piece, net)
        if action:
            rot, col = action
            shape = [list(row) for row in rot]
            lines, newb = place_piece_np(board_np.copy(), shape, col, next_piece)
            feats = get_features(newb)
            score = net.activate([float(x) for x in feats])[0]
            if score > best_score:
                best_score = score
                best = ("hold_next", next_piece, rot, col)

    # Try holding -> use stored hold piece
    if hold_piece:
        action = get_best_move_for_piece(board_np, hold_piece, net)
        if action:
            rot, col = action
            shape = [list(row) for row in rot]
            lines, newb = place_piece_np(board_np.copy(), shape, col, hold_piece)
            feats = get_features(newb)
            score = net.activate([float(x) for x in feats])[0]
            if score > best_score:
                best_score = score
                best = ("hold_swap", hold_piece, rot, col)

    return best_score, best


def get_best_move_for_piece(board_numpy, piece, net):
    best_score = -math.inf
    best_action = None
    for rotation in TETROMINOES[piece]:
        shape = np.array(rotation)
        for col in range(BOARD_WIDTH - shape.shape[1] + 1):
            if check_collision_np(board_numpy, shape, 0, col):
                continue
            lines_cleared, newb = place_piece_np(board_numpy.copy(), shape, col, piece)
            if lines_cleared == -1:
                continue
            features = get_features(newb)
            # net expects numeric input; convert to floats
            try:
                score = net.activate([float(x) for x in features])[0]
            except Exception:
                score = - (sum(features))
            if score > best_score:
                best_score = score
                best_action = (rotation, col)
    return best_action

class AIPlayer(Player):
    def __init__(self, offset_x, net=None, config=None):
        super().__init__(offset_x)
        self.net = net
        self.config = config
        self.fallback = (net is None)

    def ai_take_action_and_drop(self):
        # fallback mode
        if self.fallback or self.net is None:
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

        # NEAT mode
        board_np = create_board_numpy_from_grid(self.board.grid)
        current_piece = self.current_piece[2]
        next_piece = self.next_piece_preview[2]
        hold_piece = self.hold_piece

        # Evaluate all possible actions
        score, (action_type, chosen_piece, rotation, col) = \
            get_best_move_with_hold(board_np, current_piece, next_piece, hold_piece, self.net)

        # If the best move requires HOLDING
        if action_type == "hold_next" or action_type == "hold_swap":
            self.hold()  # performs real in-game hold

            # After hold(), update board and run regular piece placement on new current piece
            board_np = create_board_numpy_from_grid(self.board.grid)
            piece = self.current_piece[2]

            action = get_best_move_for_piece(board_np, piece, self.net)
            if action is None:
                return self.lock_piece()
            rotation, col = action
            shape = [list(row) for row in rotation]
            color = TETROMINO_COLORS[piece]
            self.current_piece = (shape, color, piece)
            self.row = 0
            self.col = col
            return self.hard_drop()

        # Otherwise: place the current piece normally
        shape = [list(row) for row in rotation]
        color = TETROMINO_COLORS[current_piece]
        self.current_piece = (shape, color, current_piece)
        self.row = 0
        self.col = col
        return self.hard_drop()


def load_ai_network(genome_path=r"C:\Users\VioletY\Desktop\MakeCode Arcade\TetrisVsAI_NEAT\best_tetris_genome.pkl", config_path=r"C:\Users\VioletY\Desktop\MakeCode Arcade\TetrisVsAI_NEAT\neat-config.txt"):
    if not HAVE_NEAT:
        print("neat python package not available. AI will be disabled.")
        return None, None
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

def draw_winner(text):
    font = pygame.font.Font(None, 80)
    surf = font.render(text, True, (255, 255, 255))
    rect = surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
    pygame.draw.rect(screen, (0, 0, 0), rect.inflate(40, 20))
    screen.blit(surf, rect)
    pygame.display.flip()

def main():
    clock = pygame.time.Clock()

    net, config = load_ai_network()  # tries to load network
    # If net is None, AI will fallback to random legal placements

    game = 0

    while game < 5:
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
                if (garbage > 0):
                    last_garbage_change = move
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
