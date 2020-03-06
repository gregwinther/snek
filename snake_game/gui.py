import numpy as np
import matplotlib.pyplot as plt


class MatplotlibGui:
    def __init__(self, title="Snake game"):
        plt.figure()
        self.title = title + " score: {0}"
        plt.title(self.title.format(0))

    def render(self, state):
        plt.clf()
        plt.title(self.title.format(state.score))
        plt.imshow(np.array(state.board))
        plt.pause(0.1)
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

        print(self.title.format(0))
        print(np.array(state.board))

    def tear_down(self):
        pass


if __name__ == "__main__":
    from snake import SnakeGame

    gui = MatplotlibGui()
    gui.render(SnakeGame(initial_length=7))
    plt.show()
