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
            action = randint(0,1) * 2 - 1
        elif vision[1] > 0:
            action = randint(0,1) - 1
        elif vision[2] > 0:
            action = randint(0,1)
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
        
    def __call__(self, state, speed):
        """Call the class. """
        time.sleep(speed)
        observation = self.generate_observation(state)
        predictions = []
        for action in range(-1, 2):
           x = torch.tensor(self.add_action_to_observation(observation, action).reshape(-1, 5))
           predictions.append(self.model(x.float()))
        return np.argmax(np.array(predictions))-1

    def generate_observation(self, state):
        """Generate observation for training. """
        # XXX: Create an array with the vision and the angle towards the food
        # return None
        
        obs = self.find_vision(state)
        angle = self.get_angle(state)
        obs.append(angle)
        return np.array(obs)

    def get_angle(self, state):
        """Get angle between snake and food. """
        # XXX: find anti-clock-wise angle between some point a and some point b
        # return None
        a = state.snake[0]
        b = state.food
        
        return np.arctan2(a[1]-b[1], a[0]-b[0])

    def get_distance(self, state):
        """Get distance between origin and coordinate b. """
        # XXX: find distance between some point b and origin
        # return None
        
        a = np.array(state.snake[0])
        b = np.array(state.food)
        return np.linalg.norm(a-b)
        
    def snake_direction(self, state):
        """Returns the moving direction of the snake. """
        # XXX: Here, you should find a unit vector of 
        # return None
        
        snake_dir = [state.snake[0][0] - state.snake[1][0],
                     state.snake[0][1] - state.snake[1][1]]
        return np.array(snake_dir)
        
    def find_vision(self, state):
        """Find vision of snake."""
        # XXX: Implement
        s     = np.array(state.snake[0])
        board = np.array(state.board)
        
        d = self.snake_direction(state)                     # directions
        f = s + d                                           # front coord
        l = s + np.array([f[0]-d[0]-d[1], f[1]+d[0]-d[1]])  # left coord
        r = s + np.array([f[0]-d[0]+d[1], f[1]-d[0]-d[1]])  # right coord
        vision = [1, 1, 1]
        if f.max() < 20 and l.max() < 20 and r.max() < 20:
            vision = [board[f[0],f[1]], 
                      board[l[0],l[1]], 
                      board[r[0],r[1]]]
        return vision

    def add_action_to_observation(self,observation, action):
        """Add action to the observation training list. """
        # XXX: append action to observation list
        # return None
        
        return np.append([action], observation)
        
        
    def generate_training_data(self,initial_games,goal_steps):
        """Generate training data based on random action. """
        training_data = []
        from tqdm import tqdm
        for i in tqdm(range(initial_games)):
            state = SnakeGame()
            prev_observation = self.generate_observation(state)
            prev_food_distance = self.get_distance(state)
            prev_score = state.score
            for j in range(goal_steps):
                action = self.generate_random_action(state)
                state = state(action)
                if state.done:
                    training_data.append([self.add_action_to_observation(prev_observation, action), -1])
                    break
                else:
                    # Scoring includes effect on food distance
                    food_distance = self.get_distance(state)
                    if state.score > prev_score or food_distance < prev_food_distance:
                        training_data.append([self.add_action_to_observation(prev_observation, action), 1])
                    else:
                        training_data.append([self.add_action_to_observation(prev_observation, action), 0])
                    prev_observation = self.generate_observation(state)
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
            # Forward Propagation
            y = self.model(x.float())
            loss = loss_func(y, t.float())
            print('epoch: ', epoch,' loss: ', loss.item()) # Zero the gradients
            optimizer.zero_grad()
            
            # Backward propagation
            loss.backward()         # perform a backward pass (backpropagation)
            optimizer.step()        # update parameters
        
    def visualise_game(self, vis_steps=500):
        game = SnakeGame(gui = True)
        done, score, snake, food, board = game.start()
        prev_observation = self.generate_observation(snake,food,board)
        for j in range(vis_steps):
            predictions = []
            for action in range(-1, 2):
               x = torch.tensor(self.add_action_to_observation(prev_observation, action).reshape(-1, 5))
               predictions.append(self.model(x.float()))
            action = np.argmax(np.array(predictions))-1
            done, score, snake, food, board  = game.step(action)
            if done:
                break
            else:
                prev_observation = self.generate_observation(snake,food,board)
        print(score)
        
        
################################################################
# CNN ENGINE
################################################################      
        
class Flatten(nn.Module):
    """Implement a flatten module."""
    def forward(self, input):
        return input.view(input.size(0), -1)
        
class CNN_Engine:
    """Snake engine controlled by a fully connected convolutional neural
    network. 
    """ 
    def __init__(self, initial_games = 1000,
                       goal_steps    = 500,
                       lr            = 2e-2,
                       max_iter      = 500):
        self.generate_training_data(initial_games, goal_steps)
        self.model_torch()
        self.train_torch(lr, max_iter)
        
    def __call__(self, state):
        predictions = []
        for action in range(-1, 2):
           x = torch.tensor(state.board)
           predictions.append(self.model(x.float()))
        return np.argmax(np.array(predictions))-1
        
    def generate_random_action(self):
        action = randint(-1, 1)
        return action
        
    def generate_training_data(self,initial_games,goal_steps):
        """Generate training data based on random action. """
        self.training_data = []
        from tqdm import tqdm
        for i in tqdm(range(initial_games)):
            game = SnakeGame()
            done, prev_score, snake, food, board = game.start()
            for j in range(goal_steps):
                action = self.generate_random_action()
                done, score, snake, food, board  = game.step(action)
                if done:
                    self.training_data.append([board, -1])
                    break
                else:
                    # Scoring includes effect on food distance
                    if score > prev_score:
                        self.training_data.append([board, 1])
                    else:
                        self.training_data.append([board, 0])
                    prev_score = score
        return self.training_data
        
    def model_torch(self):
        modules = []
        modules.append(nn.Conv2d(1, 20, kernel_size=5, stride=1, padding=2))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))
        modules.append(Flatten())
        modules.append(nn.Linear(10*10*1, 25))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Linear(25, 1))
        self.model = nn.Sequential(*modules)
        return self.model
        
    def train_torch(self, lr=1e-2, max_iter=1000):
        # Get data
        x = torch.tensor([i[0] for i in self.training_data])
        t = torch.tensor([i[1] for i in self.training_data])
        
        # Define loss and optimizer
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr) 
            
        # Train network
        for epoch in range(max_iter):
            # Forward Propagation
            y = self.model(x.float())
            loss = loss_func(y, t.float())
            print('epoch: ', epoch,' loss: ', loss.item())    # Zero the gradients
            optimizer.zero_grad()
            
            # Backward propagation
            loss.backward()         # perform a backward pass (backpropagation)
            optimizer.step()        # Update the parameters
    
if __name__ == "__main__":
    engine = DNN_Engine(initial_games=500, lr=2e-2, max_iter=500, goal_steps=500)
    from player import Player
    from gui import MatplotlibGui
    player = Player(SnakeGame(), engine, MatplotlibGui())
    player.play_game()
