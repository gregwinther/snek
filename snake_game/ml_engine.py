from random import randint
import numpy as np
import torch
import time
import torch.nn as nn
from snake import SnakeGame

################################################################
# RANDOM ENGINE
################################################################


class Random:
    """Engine based on random moves."""

    def __init__(self):
        return None

    def __call__(self, state):
        # XXX: Implement random state (-1, 0, 1)
        # return None

        return randint(-1, 1)


class RandomAssisted:
    """Engine based on random moves, but the snake should
    not crash if it can avoid it. """

    def __init__(self):
        return None

    def __call__(self, state):
        # XXX: Implement
        vision = state.vision
        if vision[0] > 0 and vision[1] > 0:
            action = -1
        elif vision[0] > 0 and vision[2] > 0:
            action = 1
        elif vision[1] > 0 and vision[2] > 0:
            action = 0
        elif vision[0] > 0:
            action = randint(0, 1) * 2 - 1
        elif vision[1] > 0:
            action = randint(0, 1) - 1
        elif vision[2] > 0:
            action = randint(0, 1)
        else:
            action = randint(-1, 1)
        return action


################################################################
# DNN ENGINE
################################################################


class DNN_Engine:
    """Snake engine controlled by a fully connected dense neural
    network. 
    """ 
    def __init__(self, initial_games = 1000,
                       goal_steps    = 500,
                       lr            = 2e-2,
                       max_iter      = 500,
                       engine = Random()):
        
        self.engine = engine
        self.model_torch()
        self.train_torch(lr, max_iter, initial_games, goal_steps)

    def generate_random_action(self, state):
        """Get random action in the interval (-1, 0, 1). """
        return self.engine(state)

    def __call__(self, state, speed=0):
        """Call the class. """
        time.sleep(speed)
        predictions = []
        for action in range(-1, 2):
            prev_observation = self.get_observation(state)
            input_data = np.append(action, prev_observation)
            x = torch.tensor(input_data.reshape(-1, 5))
            predictions.append(self.model(x.float()))
        return np.argmax(np.array(predictions))-1

    @staticmethod
    def get_angle(food):
        """Get angle between snake and food in the coordinate system of snake. """
        # XXX: find anti-clock-wise angle between some point a and some point b
        # return None
        
        return np.arctan2(food[1], food[0])

    @staticmethod
    def get_distance(state):
        """Get distance between origin and coordinate b. """
        # XXX: find distance between some point b and origin
        # return None

        a = np.array(state.snake[0])
        b = np.array(state.food)
        return np.linalg.norm(a - b)

    @staticmethod
    def snake_direction(state):
        """Returns the moving direction of the snake. """
        # XXX: Here, you should find a unit vector of
        # return None

        snake_dir = [
            state.snake[0][0] - state.snake[1][0],
            state.snake[0][1] - state.snake[1][1],
        ]
        return np.array(snake_dir)

    @staticmethod
    def get_vision(snake_head,snake_dir,snake_dir_ort,board):
        """Get vision of snake (in front, to the left and to the right)."""
        # XXX: Implement
        
        front = snake_head + snake_dir              # front coordinate
        right = snake_head - snake_dir_ort          # right coordinate
        left = snake_head + snake_dir_ort           # left coordinate
        try:
            vision = [board[front[0],front[1]], 
                      board[right[0],right[1]], 
                      board[left[0],left[1]]]
        except:
            vision = [1, 1, 1]

        return vision
        
    @staticmethod
    def transform_coord(coord,snake_head,snake_dir,snake_dir_ort):
        """Tranform coordinate. """
        
        # Define transformation matrix
        matrix = np.array([snake_dir, snake_dir_ort])
        
        # Move coordinate system such that the snake head is origin
        coord = np.array(coord) - np.array(snake_head)
        
        # Rotate coordinate system relative to the snake direction
        trans = matrix.dot(coord)
        return [int(trans[0]), int(trans[1])]
        
    def get_observation(self,state):
        """Generate observation to be used as input to
        the neural network. The next action is not known at this point,
        so we need to add it later. """
        
        # Moving direction of snake head
        snake_dir = self.snake_direction(state)
        
        # Unit vector pointing orthogonal to the moving direction
        snake_dir_ort = np.asarray(state._turn_left(snake_dir[0], snake_dir[1]))
            
        # Transform food to snake coordinates
        transform_food = self.transform_coord(state.food, state.snake[0], snake_dir, snake_dir_ort)
        
        # Snake vision
        vision = self.get_vision(state.snake[0], snake_dir, snake_dir_ort, state.board)
        
        # Angle towards the food seen from snake
        angle = self.get_angle(transform_food)
        return np.array([vision[0], vision[1], vision[2], angle])
        
    @staticmethod
    def pack_data(input_data,action,target):
        """Pack input data and target for the current move.
        should take the form:
        [(action, vision, angle), target]
        """
        # XXX: Implement
        input_data = np.append(action, input_data)
        return [input_data, target]
        
    def generate_training_data(self,initial_games,goal_steps):
        """Generate training data for the neural network 
        based on random action. """
        training_data = []
        from tqdm import tqdm
        for i in tqdm(range(initial_games)):
            state = SnakeGame()
            prev_food_distance = self.get_distance(state)
            prev_score = state.score
            prev_observation = self.get_observation(state)
            for j in range(goal_steps):
                action = self.generate_random_action(state)
                state = state(action)
                
                # We will now evaluate the performed moves, using
                # a target system where -1 means a bad move, 0 means a neutral 
                # move and 1 means a good move. 
                
                # A move is bad if the snake crashes.  
                if state.done:
                    target = -1
                    training_data.append(self.pack_data(prev_observation, action, target))
                    break
                else:
                    food_distance = self.get_distance(state)
                    
                    # A move is considered as good if the snake 
                    # gets closer to the food or eats the food. 
                    if state.score > prev_score or food_distance < prev_food_distance:
                        target = 1
                    else:
                        target = 0
                    training_data.append(self.pack_data(prev_observation, action, target))
                    prev_observation = self.get_observation(state)
                    prev_food_distance = food_distance
                    prev_score = state.score
        return training_data

    def model_torch(self):
        """Model of a dense neural network. """
        # XXX: implement a dense neural network using pytorch
        # modules = []
        # --- do something here ---
        # self.model = nn.Sequential(*modules)
        # return self.model

        modules = []
        modules.append(nn.Linear(5, 25))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(25, 1))
        self.model = nn.Sequential(*modules)
        return self.model

    def train_torch(self, lr, max_iter, initial_games, goal_steps):
        # Get data
        training_data = self.generate_training_data(initial_games, goal_steps)
        x = torch.tensor([i[0] for i in training_data]).reshape(-1, 5)
        t = torch.tensor([i[1] for i in training_data]).reshape(-1, 1)

        # Define loss and optimizer
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Train network
        for epoch in range(max_iter):
            # Forward propagation
            y = self.model(x.float())
            loss = loss_func(y, t.float())
            print("epoch: ", epoch, " loss: ", loss.item())  # Zero the gradients
            optimizer.zero_grad()

            # Backward propagation
            loss.backward()   # perform a backward pass (backpropagation)
            optimizer.step()  # update parameters

    def visualise_game(self, vis_steps=500):
        game = SnakeGame(gui = True)
        state = game.start()
        prev_observation = self.generate_observation(state)
        for j in range(vis_steps):
            predictions = []
            for action in range(-1, 2):
                x = torch.tensor(
                    self.add_action_to_observation(prev_observation, action).reshape(
                        -1, 5
                    )
                )
                predictions.append(self.model(x.float()))
            action = np.argmax(np.array(predictions)) - 1
            done, score, snake, food, board = game.step(action)
            if done:
                break
            else:
                prev_observation = self.generate_observation(snake, food, board)
        print(score)

if __name__ == "__main__":
    from player import Player
    from gui import MatplotlibGui
    
    engine = DNN_Engine(initial_games=1000, lr=2e-2, max_iter=500, goal_steps=1000)
    player = Player(SnakeGame(), engine, MatplotlibGui())
    player.play_game()
