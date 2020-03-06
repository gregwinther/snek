class Player:
    def __init__(self, game, engine, gui):
        self.game = game
        self.engine = engine
        self.gui = gui

    def play_game(self):
        while not game.done:
            self.gui.render(self.game)
            action = self.engine(self.game)
            self.game = self.game(action)

        self.gui.render(self.game)
        self.gui.tear_down()
