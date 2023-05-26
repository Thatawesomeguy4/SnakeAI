import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy
import time

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

#reset
#reward
#play(action) -> direction
#game_iteration
#is_collision

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 60
HEALTH = 50

class SnakeGameAI:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.timer = time.perf_counter()
        self.reset()
        self.hunger = HEALTH

    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.hunger = HEALTH
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.timer = time.perf_counter()

        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action):
        #increase the iteration counter to ensure no AI idling in place
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        #check if we hit a wall or if the snake is simply idling, both result in a faliure and loss of "reward" value
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
            
        # 4. place new food or just move and check to see if the worm starved to death
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.hunger = HEALTH        # set health back to max
            self._place_food()
        elif self.hunger == 0:      # game over condition for starvation
            game_over = True
            reward = -10
            return reward, game_over, self.score
        else:
            self.hunger -= 1
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        #draw hunger bar
        pygame.draw.rect(self.display, RED, pygame.Rect(self.head.x, self.head.y - 20, HEALTH, 10))
        pygame.draw.rect(self.display, (0, 128, 0), pygame.Rect(self.head.x, self.head.y - 20, HEALTH - (HEALTH - self.hunger), 10))

        text = font.render("Score: " + str(self.score), True, WHITE)
        timer = font.render(f"Time: {time.perf_counter() - self.timer:0.2f}", True, WHITE)
        self.display.blits([(text, (0, 0)), (timer, (200, 0))])

        pygame.display.flip()
        
    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if numpy.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] #no change
        elif numpy.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4 # using mod 4 ensures that if we reach index = 4 then it goes back to index = 0 since (4 mod 4) = 0. the highest valid index is 3 which is direction.up
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4 
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
            