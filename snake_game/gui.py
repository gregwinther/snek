import numpy as np
import matplotlib.pyplot as plt


class MatplotlibGui:
    def __init__(self, title="Snake game"):
        plt.figure()
        plt.title(title)

    def render(self, state):
        plt.clf()
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
