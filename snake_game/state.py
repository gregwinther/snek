import numpy as np
from random import randint, choice


BORDER = 1
BODY = 2
HEAD = 3
FOOD = -1


ACTIONS = dict(FORWARD=0, LEFT=-1, RIGHT=1)


class State:
    def __init__(self, board_width=20, board_height=20, initial_length=3):
        assert (board_height - 2) - initial_length > 0 and (
            board_width - 2
        ) - initial_length > 0
        assert initial_length > 1

        self.board_height = board_height
        self.board_width = board_width
        self.initial_length = initial_length

        self.board = self.init_board(self.board_width, self.board_height)
        self.snake = self.init_snake(
            self.initial_length, self.board_width, self.board_height
        )
        self.place_snake()
        self.food = self.place_food()

        self.done = False
        self.score = 0

    @staticmethod
    def init_board(width, height):
        # XXX: This should be constructed as an exercise
        board = [[0] * width for i in range(height)]

        for i in range(width):
            board[0][i] = BORDER
            board[height - 1][i] = BORDER

        for i in range(height):
            board[i][0] = BORDER
            board[i][width - 1] = BORDER

        return board

    @staticmethod
    def init_snake(snake_length, width, height):
        # XXX: This should be constructed as an exercise
        # Note: Use consistent ordering of rows and columns with the board

        ix_0 = randint(1, width - snake_length - 1)
        iy_0 = randint(1, height - snake_length - 1)

        horizontal = randint(0, 1) == 1
        reverse = randint(0, 1) == 1

        snake = [
            [iy_0, ix_0 + i] if horizontal else [iy_0 + i, ix_0]
            for i in range(snake_length)
        ]

        return snake if not reverse else list(reversed(snake))

    def place_snake(self):
        # XXX: This should be constructed as an exercise

        # Place head of the snake
        iy, ix = self.snake[0]
        self.board[iy][ix] = HEAD

        # Place the body of the snake
        for point in self.snake[1:]:
            iy, ix = point
            self.board[iy][ix] = BODY

    def place_food(self):
        # XXX: This should be constructed as an exercise

        ix = randint(1, self.board_width - 2)
        iy = randint(1, self.board_height - 2)

        point = [iy, ix]

        if point in self.snake:
            return self.place_food()

        self.board[iy][ix] = FOOD

        return point

    def step(self, action):
        assert action in ACTIONS.values()

        if self.done:
            return self

        self._move_snake_head(action)
        self._check_collisions()

        if self._food_eaten():
            self.score += 1
            self.place_food()
        else:
            self._remove_snake_tail()

        self.place_snake()

        return self

    def _move_snake_head(self, action):
        # XXX: Exercise
        i_0, j_0 = self.snake[0]
        i_1, j_1 = self.snake[1]

        delta_i, delta_j = i_0 - i_1, j_0 - j_1

        if action == ACTIONS["LEFT"]:
            delta_i, delta_j = self._turn_left(delta_i, delta_j)
        elif action == ACTIONS["RIGHT"]:
            delta_i, delta_j = self._turn_right(delta_i, delta_j)

        new_head = [i_0 + delta_i, j_0 + delta_j]
        self.snake.insert(0, new_head)

    def _turn_left(self, delta_i, delta_j):
        # XXX: Exercise
        return delta_j, -delta_i

    def _turn_right(self, delta_i, delta_j):
        # XXX: Exercise
        return -delta_j, delta_i

    def _check_collisions(self):
        # XXX: Exercise
        i_0, j_0 = self.snake[0]
        if self.board[i_0][j_0] > 0:
            self.done = True

    def _food_eaten(self):
        # XXX: Exercise
        return self.snake[0] == self.food

    def _remove_snake_tail(self):
        # XXX: Exercise
        i, j = self.snake.pop()
        self.board[i][j] = 0


if __name__ == "__main__":
    # Test initialization
    state = State(board_width=6, board_height=6, initial_length=3)
    print("\033c", end="")
    print(np.array(state.board))

    import time

    time.sleep(0.5)

    for i in range(10):
        state.step(choice(list(ACTIONS.values())))
        print("\033c", end="")
        print(np.array(state.board))
        time.sleep(0.5)
