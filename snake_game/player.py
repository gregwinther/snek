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
    from gui import MatplotlibGui, TerminalGui, YeetTerminalGui

    snake_game = SnakeGame()
    keyboard_listener = KeyboardListener(snake_game)

    player = Player(snake_game, keyboard_listener, YeetTerminalGui(), speed=7)
    player.play_game()
