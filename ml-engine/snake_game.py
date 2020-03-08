# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from random import randint, seed
import numpy as np

seed(30)

class SnakeGame:
    def __init__(
        self,
        board_width=20,
        board_height=20,
        gui=False,
        initial_length=3,
        obs_pos=[[1, 0], [0, -1], [0, 1]],
    ):
        self.snake = []
        self.score = 0
        self.done = False
        self.board_width = board_width
        self.board_height = board_height
        self.gui = gui
        self.initial_length = initial_length
        self.board = [[0] * board_width for i in range(board_height)]
        # Create borders
        for i in range(self.board_width):
            self.board[i][0] = 1
            self.board[i][self.board_height - 1] = 1
        for i in range(self.board_height):
            self.board[0][i] = 1
            self.board[self.board_width - 1][i] = 1
        # Observation positions
        self.obs_pos = obs_pos
        self.num_obs_pos = len(self.obs_pos)

    def start(self):
        if self.gui:
            self.render_init()
        self.snake_init()
        self.generate_food()
        if self.gui:
            self.render()
        return self.generate_observations()

    def snake_init(self):
        x = randint(5, self.board_width - 5)
        y = randint(5, self.board_height - 5)
        self.snake = []
        vertical = randint(0, 1) == 0
        for i in range(self.initial_length):
            point = [x + i, y] if vertical else [x, y + i]
            self.snake.insert(0, point)
            if i == self.initial_length - 1:
                val = 3  # head
            else:
                val = 2  # body
            self.board[point[0]][point[1]] = val

    def generate_food(self):
        food = []
        while food == []:
            food = [
                randint(3, self.board_width - 3),
                randint(3, self.board_height - 3),
            ]
            if food in self.snake:
                food = []
        self.food = food
        self.board[food[0]][food[1]] = -1

    def render_init(self):
        _ = plt.figure(1)
        plt.title("Snake game")
        self._running = True

    def render(self):
        plt.clf()
        plt.imshow(np.array(self.board))
        plt.title("Score: " + str(self.score))
        plt.axis('off')
        plt.pause(0.1)
        plt.draw()

    def step(self, key):
        # -1 - LEFT
        #  0 - FORWARD
        #  1 - RIGHT
        #print(self.board)
        #print(self.food)
        #print(self.snake)
        if self.done == True:
            self.end_game()
        self.create_new_point_in_snake(key)
        self.check_collisions()
        self.place_new_point_on_board()
        if self.food_eaten():
            self.score += 1
            self.generate_food()
        else:
            self.remove_last_point()
        if self.gui:
            self.render()
        return self.generate_observations()

    def create_new_point_in_snake(self, key):
        snake0 = self.snake[0]
        snake1 = self.snake[1]
        snake_dir = [snake0[0] - snake1[0], snake0[1] - snake1[1]]
        move = snake_dir
        if key == -1:
            move = self.turn_to_the_left(move)
        elif key == 1:
            move = self.turn_to_the_right(move)
        new_point = [snake0[0] + move[0], snake0[1] + move[1]]
        self.snake.insert(0, new_point)

    def place_new_point_on_board(self):
        new_point = self.snake[0]
        self.board[new_point[0]][new_point[1]] = 3  # head
        old_point = self.snake[1]
        self.board[old_point[0]][old_point[1]] = 2  # body

    def turn_to_the_left(self, vector):
        return [-vector[1], vector[0]]

    def turn_to_the_right(self, vector):
        return [vector[1], -vector[0]]

    def remove_last_point(self):
        x, y = self.snake.pop()
        self.board[x][y] = 0

    def food_eaten(self):
        return self.snake[0] == self.food

    def check_collisions(self):
        x = self.snake[0][0]
        y = self.snake[0][1]
        if self.board[x][y] > 0:
            self.done = True

    def generate_observations(self):
        # Returns all positions in coordinate system relative
        # to the snake head
        origin = self.snake[0]
        snake_dir = [
            self.snake[0][0] - self.snake[1][0],
            self.snake[0][1] - self.snake[1][1],
        ]
        snake_dir_ort = self.turn_to_the_left(snake_dir)  # Normal to snake_dir
        # Find matrix for coordinate transformation
        matrix = np.array([snake_dir, snake_dir_ort])
        
        # Tranform snake
        transform_snake = self.snake.copy()
        for i in range(len(transform_snake)):
            pos = np.array(transform_snake[i]) - np.array(origin)
            trans = matrix.dot(pos)
            transform_snake[i] = [int(trans[0]), int(trans[1])]
            
        # Transform food to snake coordinates
        pos = np.array(self.food) - np.array(origin)
        trans = matrix.dot(pos)
        transform_food = [int(trans[0]), int(trans[1])]
        # Find transform of board up to a given distance from the origin
        # Start by (2n+1)x(2n+1) size around origin
        # Fill with border outside of the board size
        
        # Snake vision
        transform_board = []

        for i in range(len(self.obs_pos)):
            ix = self.obs_pos[i][0]
            iy = self.obs_pos[i][1]
            x = ix * snake_dir[0] + iy * snake_dir_ort[0] + origin[0]
            y = ix * snake_dir[1] + iy * snake_dir_ort[1] + origin[1]
            if (
                x >= 0
                and x < self.board_width
                and y >= 0
                and y < self.board_height
            ):
                boardval = self.board[x][y]
            else:
                boardval = 1  # Mark outside as a wall
            if boardval > 1:  # Transform to 0 to 1 range
                boardval = 1
            transform_board.append(boardval)

        return (
            self.done,
            self.score,
            transform_snake,
            transform_food,
            transform_board,
        )

    def render_destroy(self):
        plt.clf()

    def end_game(self):
        if self.gui:
            self.render_destroy()
        raise Exception("Game over")


if __name__ == "__main__":
    game = SnakeGame(gui=True)
    done, score, snake, food, board = game.start()
    while True:
        print(board)
        if board[0] > 0 and board[1] > 0:
            action = -1
        elif board[0] > 0 and board[2] > 0:
            action = 1
        elif board[1] > 0 and board[2] > 0:
            action = 0
        elif board[0] > 0:
            action = randint(0,1) * 2 - 1
        elif board[1] > 0:
            action = randint(0,1) - 1
        elif board[2] > 0:
            action = randint(0,1)
        else:
            action = randint(-1, 1)
        print(action)
        done, score, snake, food, board = game.step(action)
    print(game.score)
    game.render_destroy()
