import numpy as np
from random import randint


BORDER = 1
BODY = 2
HEAD = 3
FOOD = -1


class State:
    def __init__(self, board_width=20, board_height=20, initial_length=3):
        assert (board_height - 2) - initial_length > 0 and (
            board_width - 2
        ) - initial_length > 0

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


if __name__ == "__main__":
    # Test initialization
    state = State(board_width=6, board_height=6, initial_length=3)
    print(np.array(state.board))
    print(state.snake)
