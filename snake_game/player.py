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
    import time

    time.sleep(speed)
    return choice(list(game.ACTIONS.values()))


if __name__ == "__main__":
    from snake import SnakeGame
    from gui import MatplotlibGui, TerminalGui

    player = Player(SnakeGame(), random_engine, TerminalGui(), speed=7)
    player.play_game()
