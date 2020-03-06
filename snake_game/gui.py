import numpy as np
import matplotlib.pyplot as plt


class NoGui:
    def __init__(self, title="Snake game"):
        self.title = title + " score: {0}"

    def render(self, state):
        print(self.title.format(state.score))

    def tear_down(self):
        pass


class MatplotlibGui:
    def __init__(self, title="Snake game"):
        plt.figure()
        self.title = title + " score: {0}"
        plt.title(self.title.format(0))

    def render(self, state):
        plt.clf()
        plt.title(self.title.format(state.score))
        plt.imshow(np.array(state.board))
        plt.pause(0.01)
        plt.draw()

    def tear_down(self):
        plt.clf()


class TerminalGui:
    def __init__(self, title="Snake game"):
        self.title = title + " score: {0}"
        self._clear_terminal()

    def _clear_terminal(self):
        print("\033c", end="")

    def render(self, state):
        self._clear_terminal()

        print(self.title.format(state.score))
        board = np.array(state.board, dtype=object)
        board[board == 0] = " "
        board[board == 1] = chr(9671)
        board[board == 2] = chr(9632)
        board[board == 3] = chr(9632)
        board[board == -1] = chr(9679)
        for row in board:
            for elem in row:
                print(elem, end=" ")
            print()

    def tear_down(self):
        pass


class YeetTerminalGui(TerminalGui):
    def render(self, state):
        self._clear_terminal()

        print(self.title.format(state.score))

        board = np.array(state.board, dtype=object)
        board[board == 0] = chr(int("1f331", 16))
        board[board == 1] = chr(int("1f5fb", 16))
        board[board == 2] = chr(int("1f534", 16))
        board[board == 3] = chr(int("1f923", 16))
        board[board == -1] = chr(int("1f4a9", 16))
        for row in board:
            for elem in row:
                print(elem, end=" ")
            print()


if __name__ == "__main__":
    from snake import SnakeGame

    gui = MatplotlibGui()
    gui.render(SnakeGame(initial_length=7))
    plt.show()
