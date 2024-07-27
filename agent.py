import torch
import random
import numpy as np
from collections import deque
from model.model import Linear_QNet, QTrainer

class Agent:
    def __init__(self):
        self.max_memory = 100000
        self.batch_size = 32
        self.lr = 0.1
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.1 # discount rate
        self.memory = deque(maxlen=self.max_memory) # popleft()
        self.model = Linear_QNet(1)
        self.trainer = QTrainer(self.model, lr=self.lr, gamma=self.gamma)

    def get_state(self, game):
        state = game.state_arr
        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if self.max_memory is reached

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 100 - self.n_games
        final_move = [0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 1)
            final_move[0] = move
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            if(prediction > 0.5):
                final_move[0] = 1
            else:
                final_move[0] = 0

        return final_move

