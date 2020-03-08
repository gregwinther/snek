from random import randint, choice


class SnakeGame:
    BORDER = 1
    BODY = 2
    HEAD = 3
    FOOD = -1

    ACTIONS = dict(FORWARD=0, LEFT=-1, RIGHT=1)

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
        self._place_snake()
        self.food = self._place_food()

        self.done = False
        self.score = 0

    @staticmethod
    def init_board(width, height):
        """Static method creating a board, and padding all edges with a border.

        Parameters
        ----------
        width : int
            The width of the board, i.e., the number of columns.
        height : int
            The height of the board, i.e., the number of rows.

        Returns
        -------
        list
            Empty board with filled edges.
        """
        pass

    @staticmethod
    def init_snake(snake_length, width, height):
        """Static method initializing the snake as a list with coordinates of
        the head and the body of the snake. This function should find a random
        location on the board (within the borders of the board) for the head of
        the snake. Next it should add elements for the body in a straight line
        from the head. Optionally the head could be placed vertically or
        horizontally on the board. Also, it could optionally be "reversed".

        Parameters
        ----------
        snake_length : int
            The length of snake, head included. Note that this length needs to
            fit on the board.
        width : int
            The width of the board, i.e., the number of columns.
        height : int
            The height of the board, i.e., the number of rows.

        Returns
        -------
        list
            A list with the coordinates of the snake. The first element (the
            first coordinate) is interpreted as the head of the snake.
        """
        pass

    def _place_snake(self):
        """Class method "placing" the snake on the board. This is done by using
        the coordinates of the snake as indices on the board, and setting the
        values of these elements to ``self.HEAD`` for the head of the snake, and
        ``self.BODY`` for the body of the snake.
        """
        pass

    def _place_food(self):
        """Class method placing food on the board. To do this you need to find
        a random point on the board that is not a border nor inside the snake.
        The first condition can be achieved by drawing valid random numbers, but
        the latter condition must be tested by checking if the food is inside the
        snake.

        Returns
        -------
        list
            The coordinates of the newly placed food.
        """
        pass

    def __call__(self, action):
        assert action in self.ACTIONS.values()

        # Check if we are done
        if self.done:
            return self

        # Move the snake one step
        self._move_snake_head(action)
        # Check if we have collided
        self._check_collisions()

        # Check if the food has been eaten
        if self._food_eaten():
            # Increase the score and place new food
            self.score += 1
            self.food = self._place_food()
        else:
            # If no food was eaten, remove last point in the snake and from the
            # board
            self._remove_snake_tail()

        # Place the updated snake on the board
        self._place_snake()

        return self

    def _move_snake_head(self, action):
        """Class method performing the specified action and creating a new head
        for the snake. This is done by finding the change in position of the
        current head. Then, depending on the action, a new coordinate for the
        head should be inserted at the beginning of the ``self.snake``-list.

        Parameters
        ----------
        action : int
            A valid action that can be found in the
            ``SnakeGame.Actions``-dictionary.
        """
        pass

    def _turn_left(self, delta_i, delta_j):
        """Class method taking in the velocity of the snake, and returning the
        new velocity by rotating the velocity vector by 90 degrees to the left.

        Parameters
        ----------
        delta_i : int
            Speed of the snake in the height-direction, i.e., along the rows of
            the board.
        delta_j : int
            Speed of the snake in the width-direction, i.e., along the columns
            of the board.

        Returns
        -------
        tuple, list
            The velocity turned 90 degrees to the left.
        """
        pass

    def _turn_right(self, delta_i, delta_j):
        """Class method taking in the velocity of the snake, and returning the
        new velocity by rotating the velocity vector by 90 degrees to the right.

        Parameters
        ----------
        delta_i : int
            Speed of the snake in the height-direction, i.e., along the rows of
            the board.
        delta_j : int
            Speed of the snake in the width-direction, i.e., along the columns
            of the board.

        Returns
        -------
        tuple, list
            The velocity turned 90 degrees to the right.
        """
        pass

    def _check_collisions(self):
        """Class method checking if the head of the snake has hit a border or
        its own body. If a collision has occured, ``self.done`` should be set to
        ``True``.
        """
        pass

    def _food_eaten(self):
        """Class method checking if the food has been eaten by the snake. It is
        enough to check if the coordinates of the head of the snake is the same
        as the coordinates of the food.

        Returns
        -------
        bool
            Whether or not the food has been eaten. ``True`` denotes that the
            food has been eaten.
        """
        pass

    def _remove_snake_tail(self):
        """Class method removing the tail, i.e., the last coordinate of the
        snake, from ``self.snake`` and the board.
        """
        pass


if __name__ == "__main__":
    import numpy as np
    import time

    # Test initialization
    state = SnakeGame(board_width=10, board_height=10, initial_length=3)
    print("\033c", end="")
    print(np.array(state.board))

    time.sleep(0.5)

    for i in range(10):
        state(choice(list(state.ACTIONS.values())))
        print("\033c", end="")
        print(np.array(state.board))
        time.sleep(0.5)
