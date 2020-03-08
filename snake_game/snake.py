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

        >>> import numpy as np
        >>> board = SnakeGame.init_board(6, 6)
        >>> print(np.array(board))
        [[1 1 1 1 1 1]
         [1 0 0 0 0 1]
         [1 0 0 0 0 1]
         [1 0 0 0 0 1]
         [1 0 0 0 0 1]
         [1 1 1 1 1 1]]
        """

        board = [[0] * width for i in range(height)]

        for i in range(width):
            board[0][i] = SnakeGame.BORDER
            board[height - 1][i] = SnakeGame.BORDER

        for i in range(height):
            board[i][0] = SnakeGame.BORDER
            board[i][width - 1] = SnakeGame.BORDER

        return board

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

        >>> from random import seed
        >>> seed(2020)
        >>> snake = SnakeGame.init_snake(3, 7, 7)
        >>> print(len(snake))
        3
        >>> for coord in snake:
        ...     print(len(coord))
        2
        2
        2
        """
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

    def _place_snake(self):
        """Class method "placing" the snake on the board. This is done by using
        the coordinates of the snake as indices on the board, and setting the
        values of these elements to ``self.HEAD`` for the head of the snake, and
        ``self.BODY`` for the body of the snake.

        >>> import numpy as np
        >>> sg = SnakeGame(10, 10, 3)
        >>> np.sum(np.array(sg.board) == sg.HEAD)
        1
        >>> np.sum(np.array(sg.board) == sg.BODY)
        2
        """
        # Place head of the snake
        iy, ix = self.snake[0]
        self.board[iy][ix] = self.HEAD

        # Place the body of the snake
        for point in self.snake[1:]:
            iy, ix = point
            self.board[iy][ix] = self.BODY

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

        >>> sg = SnakeGame(10, 10, 3)
        >>> len(sg.food)
        2
        """
        ix = randint(1, self.board_width - 2)
        iy = randint(1, self.board_height - 2)

        point = [iy, ix]

        while point in self.snake:
            ix = randint(1, self.board_width - 2)
            iy = randint(1, self.board_height - 2)

            point = [iy, ix]

        self.board[iy][ix] = self.FOOD

        return point

    def __call__(self, action):
        assert action in self.ACTIONS.values()

        if self.done:
            return self

        self._move_snake_head(action)
        self._check_collisions()

        if self._food_eaten():
            self.score += 1
            self.food = self._place_food()
        else:
            self._remove_snake_tail()

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

        >>> sg = SnakeGame(10, 10, 2)
        >>> sg.snake = [[5, 5], [5, 4]]
        >>> sg._place_snake()
        >>> sg._move_snake_head(sg.ACTIONS["FORWARD"])
        >>> sg.snake
        [[5, 6], [5, 5], [5, 4]]
        >>> sg._move_snake_head(sg.ACTIONS["LEFT"])
        >>> sg.snake
        [[4, 6], [5, 6], [5, 5], [5, 4]]
        >>> sg._move_snake_head(sg.ACTIONS["RIGHT"])
        >>> sg.snake
        [[4, 7], [4, 6], [5, 6], [5, 5], [5, 4]]
        """
        i_0, j_0 = self.snake[0]
        i_1, j_1 = self.snake[1]

        delta_i, delta_j = i_0 - i_1, j_0 - j_1

        if action == self.ACTIONS["LEFT"]:
            delta_i, delta_j = self._turn_left(delta_i, delta_j)
        elif action == self.ACTIONS["RIGHT"]:
            delta_i, delta_j = self._turn_right(delta_i, delta_j)

        new_head = [i_0 + delta_i, j_0 + delta_j]
        self.snake.insert(0, new_head)

    @staticmethod
    def _turn_left(delta_i, delta_j):
        """Static method taking in the velocity of the snake, and returning the
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
        list
            The velocity turned 90 degrees to the left.

        >>> SnakeGame._turn_left(1, 0)
        [0, 1]
        >>> SnakeGame._turn_left(0, 1)
        [-1, 0]
        """
        return [-delta_j, delta_i]

    @staticmethod
    def _turn_right(delta_i, delta_j):
        """Static method taking in the velocity of the snake, and returning the
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

        >>> SnakeGame._turn_right(1, 0)
        [0, -1]
        >>> SnakeGame._turn_right(0, 1)
        [1, 0]
        """
        return [delta_j, -delta_i]

    def _check_collisions(self):
        """Class method checking if the head of the snake has hit a border or
        its own body. If a collision has occured, ``self.done`` should be set to
        ``True``.

        >>> sg = SnakeGame(10, 10, 3)
        >>> sg.done
        False
        >>> sg.snake = [[0, 2], [1, 2], [2, 2]]
        >>> sg._place_snake()
        >>> sg._check_collisions()
        >>> sg.done
        True
        """
        i_0, j_0 = self.snake[0]

        if self.board[i_0][j_0] > 0:
            self.done = True

    def _food_eaten(self):
        """Class method checking if the food has been eaten by the snake. It is
        enough to check if the coordinates of the head of the snake is the same
        as the coordinates of the food.

        Returns
        -------
        bool
            Whether or not the food has been eaten. ``True`` denotes that the
            food has been eaten.

        >>> sg = SnakeGame(10, 10, 3)
        >>> sg.snake = [[5, 5], [5, 4], [5, 3]]
        >>> sg._place_snake()
        >>> sg.food = [5, 6]
        >>> sg._move_snake_head(sg.ACTIONS["FORWARD"])
        >>> sg.snake
        [[5, 6], [5, 5], [5, 4], [5, 3]]
        >>> sg._food_eaten()
        True
        """

        return self.snake[0] == self.food

    def _remove_snake_tail(self):
        """Class method removing the tail, i.e., the last coordinate of the
        snake, from ``self.snake`` and the board.

        >>> sg = SnakeGame(10, 10, 3)
        >>> sg.snake = [[5, 5], [5, 4], [5, 3]]
        >>> sg._place_snake()
        >>> sg.board[5][3] == sg.BODY
        True
        >>> sg._remove_snake_tail()
        >>> sg.board[5][3] == 0
        True
        >>> sg.snake
        [[5, 5], [5, 4]]
        """

        i, j = self.snake.pop()
        self.board[i][j] = 0


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
