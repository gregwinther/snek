from random import randint
import numpy as np
import math
import torch
import torch.nn as nn
from snake_game import SnakeGame

################################################################
# RANDOM ENGINE
################################################################

class Random:
    """Engine based on random moves."""
    def __init__(self):
        return None
        
    def predict(self, state):
        return randint(-1, 1)



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
                       max_iter      = 500):
        self.generate_training_data(initial_games, goal_steps)
        self.model_torch()
        self.train_torch(lr, max_iter)
        
    def predict(self, state):
        observation = self.generate_observation(state.snake, state.food, state.board)
        predictions = []
        for action in range(-1, 2):
           x = torch.tensor(self.add_action_to_observation(observation, action).reshape(-1, 5))
           predictions.append(self.model(x.float()))
        return np.argmax(np.array(predictions))-1
        
    #def generate_random_action(self):
    #    action = randint(-1, 1)
    #    return action
        
    def generate_random_action(self, board, assist=False):
        # FRONT: board[0] -> action 0
        # RIGHT: board[1] -> action 1
        # LEFT:  board[2] -> action -1 
        if assist:
            if board[0] > 0 and board[1] > 0:
                action = -1
            elif board[0] > 0 and board[2] > 0:
                action = 1
            elif board[1] > 0 and board[2] > 0:
                action = 0
            elif board[0] > 0:
                action = randint(0,1) * 2 - 1
            elif board[1] > 0:
                action = randint(0,1) - 1
            elif board[2] > 0:
                action = randint(0,1)
            else:
                action = randint(-1, 1)
        else:
            action = randint(-1, 1)
        return action

    def generate_observation(self,snake,food,board):
        obs = board.copy()
        angle = self.get_angle(food)
        obs.append(angle)
        return np.array(obs)

    def get_angle(self,b):
        """Get angle between origin coordinate b. """
        return math.atan2(b[1], b[0])/(np.linalg.norm(b)*math.pi)

    def get_distance(self,b):
        """Get distance between origin and coordinate b. """
        return np.linalg.norm(b)

    def add_action_to_observation(self,observation, action):
        return np.append([action], observation)
        
    def generate_training_data(self,initial_games, goal_steps, print_interval=1000):
        """Generate training data based on random action. """
        self.training_data = []
        from tqdm import tqdm
        for i in tqdm(range(initial_games)):
            game = SnakeGame()
            done, prev_score, snake, food, board = game.start()
            prev_observation = self.generate_observation(snake, food, board)
            prev_food_distance = self.get_distance(food)
            for j in range(goal_steps):
                action = self.generate_random_action(board)
                done, score, snake, food, board  = game.step(action)
                if done:
                    self.training_data.append([self.add_action_to_observation(prev_observation, action), -1])
                    break
                else:
                    # Scoring includes effect on food distance
                    food_distance = self.get_distance(food)
                    if score > prev_score or food_distance < prev_food_distance:
                        self.training_data.append([self.add_action_to_observation(prev_observation, action), 1])
                    else:
                        self.training_data.append([self.add_action_to_observation(prev_observation, action), 0])
                    prev_observation = self.generate_observation(snake,food,board)
                    prev_food_distance = food_distance
                    prev_score = score
        return self.training_data
        
    def model_torch(self):
        modules = []
        modules.append(nn.Linear(5, 25))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(25, 1))
        self.model = nn.Sequential(*modules)
        return self.model
        
    def train_torch(self, lr=1e-2, max_iter=1000):
        # Get data
        x = torch.tensor([i[0] for i in self.training_data]).reshape(-1, 5)
        t = torch.tensor([i[1] for i in self.training_data]).reshape(-1, 1)
        
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
        points = len(snake) - 3
        return points
        
        
        
################################################################
# CNN ENGINE
################################################################        
        
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
        
    def predict(self, state):
        predictions = []
        for action in range(-1, 2):
           x = torch.tensor(state.board)
           predictions.append(self.model(x.float()))
        return np.argmax(np.array(predictions))-1
        
    def generate_random_action(self):
        action = randint(-1, 1)
        return action
        
    def generate_training_data(self,initial_games, goal_steps, print_interval=1000):
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
        self.model = nn.Sequential(*modules)
        return self.model
        
    def train_torch(self, lr=1e-2, max_iter=1000):
        # Get data
        x = torch.tensor([i[0] for i in self.training_data]).reshape(-1, 5)
        t = torch.tensor([i[1] for i in self.training_data]).reshape(-1, 1)
        
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
        points = len(snake) - 3
        return points
    
if __name__ == "__main__":
    engine = DNN_Engine(initial_games=1000, lr=2e-2, max_iter=500, goal_steps=1000)
    engine.visualise_game()
