import pygame
import random
import copy
from collections import deque

pygame.init()

CELL_SIZE = 30
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
MARGIN = 2
PREVIEW_WIDTH = 6
PREVIEW_HEIGHT = 4
SIDE_WIDTH = 150
SCREEN_WIDTH = 2.5 * (BOARD_WIDTH * (CELL_SIZE + MARGIN) + SIDE_WIDTH)
SCREEN_HEIGHT = BOARD_HEIGHT * (CELL_SIZE + MARGIN)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("2-Player Tetris")

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
        font = pygame.font.Font(None, 80)
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
        self.draw_garbage(next_x, 2900, rect)



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

game = 0
while game < 1:
    # Game loop
    game += 1
    clock = pygame.time.Clock()
    player1 = Player(90)
    player2 = Player(450)
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
        
        # Player 1
        if not player1.game_over:
            garbage = 0
            handle_das(player1, pygame.K_a, pygame.K_d, keys, prev_keys)
            if keys[pygame.K_w] and not prev_keys[pygame.K_w]: player1.rotate()
            if keys[pygame.K_s] and move%3 == 0: garbage += player1.soft_drop()
            if keys[pygame.K_q] and not prev_keys[pygame.K_q]: player1.hold()
            if keys[pygame.K_e] and not prev_keys[pygame.K_e]: garbage += player1.hard_drop()

            # if (False):
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
            pygame.time.wait(3000)
            draw_winner("Player 2 Wins!")
            pygame.time.wait(3000)
            running = False

        # Player 2
        if not player2.game_over:
            garbage = 0
            handle_das(player2, pygame.K_LEFT, pygame.K_RIGHT, keys, prev_keys)
            if keys[pygame.K_UP] and not prev_keys[pygame.K_UP]: player2.rotate()
            if keys[pygame.K_DOWN] and move%3 == 0: garbage += player2.soft_drop()
            if keys[pygame.K_SLASH] and not prev_keys[pygame.K_SLASH]: player2.hold()
            if keys[pygame.K_RSHIFT] and not prev_keys[pygame.K_RSHIFT]: garbage += player2.hard_drop()

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
            pygame.time.wait(3000)
            draw_winner("Player 1 Wins!")
            pygame.time.wait(3000)
            running = False

        if (move - last_garbage_change > 240):
            tot_garb = abs(player1.garbage_queue - player2.garbage_queue)
            if (player1.garbage_queue > player2.garbage_queue):
                player1.garbage_queue = tot_garb
                player2.garbage_queue = 0
                player1.apply_garbage()
            elif (player2.garbage_queue > player1.garbage_queue):
                player2.garbage_queue = tot_garb
                player1.garbage_queue = 0
                player2.apply_garbage()

        player1.draw()
        player2.draw()

        pygame.display.flip()
        clock.tick(60)
        move += 1

        prev_keys = keys

pygame.quit()