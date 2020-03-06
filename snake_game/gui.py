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


if __name__ == "__main__":
    from state import State

    gui = MatplotlibGui()
    gui.render(State(initial_length=7))
    plt.show()
