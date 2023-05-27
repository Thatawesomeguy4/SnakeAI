import torch
import random
import numpy
from game import SnakeGameAI, Direction, Point
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0                                                # counts the number of elapsed games
        self.epsilon = 0                                                # parameter to control randomness
        self.gamma = 0.9                                                # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)                          # popleft() if memory is exceeded
        self.model = Linear_QNet(11, 512, 3)                            # input size,
                                                                        # define hidden size,and define output
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)    # TODO

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger Straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger Right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger Left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food Location
            game.food.x < game.head.x,      # food is to the left
            game.food.x > game.head.x,      # food is to the right
            game.food.y < game.head.y,      # food is above
            game.food.y > game.head.y,       # food is below

        ]

        return numpy.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, game_over_state):
        self.memory.append((state, action, reward, next_state, game_over_state)) # popleft if MAX_MEMORY is reached.

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)    # returns a list of tuples if we already have 1000 samples in memory
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, game_over_states = zip(*mini_sample)         # zip extracts all of the variables for us until input is exhausted
        self.trainer.train_step(states, actions, rewards, next_states, game_over_states)


    def train_short_memory(self, state, action, reward, next_state, game_over_state):
        self.trainer.train_step(state, action, reward, next_state, game_over_state)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games # the more games we have, the smaller our epsilon value becomes
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon: # the smaller the epsilon value gets, the less we randomly explore and the more we rely on the trained model
            move = random.randint(0, 2)
            final_move[move] = 1
        else: # make a move based on our trained model instead of a random move
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    plot_time = []
    plot_mean_times = []
    total_score = 0
    total_time = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True: # this represents the game loop which should run infinitely until we say "stop"
        # get the old state of the game (current state)
        state_old = agent.get_state(game)

        # get old state's move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, game_over_state, score, time = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, game_over_state)

        # remember
        agent.remember(state_old, final_move, reward, state_new, game_over_state)

        if game_over_state:
            # train the long memory (trains over all completed games and moves to this point)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            plot_time.append(time)
            total_time += time
            mean_time = total_time / agent.n_games
            plot_mean_times.append(mean_time)
            plot(plot_scores, plot_mean_scores, plot_time, plot_mean_times)


if __name__ == '__main__':
    train()
