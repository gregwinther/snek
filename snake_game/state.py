import numpy as np
from random import randint


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
            board[0][i] = 1
            board[height - 1][i] = 1

        for i in range(height):
            board[i][0] = 1
            board[i][width - 1] = 1

        return board

    @staticmethod
    def init_snake(snake_length, width, height):
        # XXX: This should be constructed as an exercise
        x_0 = randint(1, width - snake_length - 2)
        y_0 = randint(1, height - snake_length - 2)

        horizontal = randint(0, 1) == 1

        return [
            [x_0 + i, y_0] if horizontal else [x_0, y_0 + i]
            for i in range(snake_length, 0, -1)
        ]

    def place_snake(self):
        pass

    def place_food(self):
        # XXX: This should be constructed as an exercise
        pass


if __name__ == "__main__":
    # Test initialization
    state = State(board_width=10, board_height=6, initial_length=3)
    print(np.array(state.board))
    print(state.snake)
