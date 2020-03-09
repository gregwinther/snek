from pynput.keyboard import Key, Listener
import threading
import time

SPEED = {i: 0.05 * i for i in range(0, 10)}


class Player:
    def __init__(self, game, engine, gui, speed=4):
        self.game = game
        self.engine = engine
        self.gui = gui
        self.speed = SPEED[speed]

    def play_game(self):
        while not self.game.done:
            self.gui.render(self.game)
            action = self.engine(self.game, self.speed)
            self.game = self.game(action)

        self.gui.render(self.game)
        self.gui.tear_down()


def random_engine(game, speed):
    from random import choice

    time.sleep(speed)
    return choice(list(game.ACTIONS.values()))


def best_engine(game, speed):
    # Note: This engine only works for even board widths!
    # It also sometimes crashes if the snake is placed alongside the border.
    time.sleep(speed)

    action = game.ACTIONS["FORWARD"]

    y_0, x_0 = game.snake[0]
    y_1, x_1 = game.snake[1]

    right_edge = game.board_width - 2
    bottom_edge = game.board_height - 2

    # At the top
    if y_0 == 1:
        action = game.ACTIONS["RIGHT"]

    # At left edge
    elif x_0 == 1 and x_1 > 1:
        action = game.ACTIONS["RIGHT"]

    # At the right edge
    elif x_0 == right_edge and x_1 < right_edge:
        action = game.ACTIONS["RIGHT"]

    # At the bottom
    elif y_0 == bottom_edge - 1:
        # Anywhere, but in bottom right corner
        if not x_0 == right_edge and x_0 > 1:
            action = game.ACTIONS["LEFT"]

    # Bottom right corner
    elif y_0 == bottom_edge and x_0 == right_edge:
        action = game.ACTIONS["RIGHT"]

    return action


class KeyboardListener:
    def __init__(self, game):
        self.game = game
        self.action_list = []

        listener = Listener(on_press=self.on_press)
        listener.start()

    def on_press(self, key):

        if key == Key.left:
            self.action_list.append(self.game.ACTIONS["LEFT"])

        if key == Key.right:
            self.action_list.append(self.game.ACTIONS["RIGHT"])

        if key == Key.esc:
            self.game.done = True

    def __call__(self, game, speed):
        time.sleep(speed)

        action = self.game.ACTIONS["FORWARD"]

        if len(self.action_list) > 0:
            action = self.action_list.pop(0)

        return action


if __name__ == "__main__":
    from snake import SnakeGame
    from gui import MatplotlibGui, TerminalGui, YeetTerminalGui, NoGui

    snake_game = SnakeGame()
    keyboard_listener = KeyboardListener(snake_game)

    player = Player(snake_game, best_engine, NoGui(), speed=0)
    player.play_game()
