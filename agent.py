import torch
import random
import numpy as np
from collections import deque
from model.model import Linear_QNet, QTrainer
from game.flappybirdAI import FlappyBirdAI
from model.helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(7, 32, 64, 1)
        # self.model = self.model.load_state_dict(torch.load("model.pth"))
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        # Position of bird
        bird_pos = game.bird_pos
        point_up = [bird_pos[0]+10,bird_pos[1]-10]
        point_down = [bird_pos[0]+10,bird_pos[1]+10]
        point_front = [bird_pos[0]+10,bird_pos[1]]

        dir_up = game.direction == 1
        dir_down = game.direction == 0
        bird = [(game.bird).copy(), (game.bird).copy(), (game.bird).copy()] # 0: front, 1: up, 2: down
        bird[0].x = bird[0].x + 10
        bird[1].x = bird[1].x + 10
        bird[1].y = bird[1].y - 10
        bird[2].x = bird[2].x + 10
        bird[2].y = bird[2].y + 10

        state = [
            # Danger in front
            (bird[0].colliderect(game.upper_pipe) or bird[0].colliderect(game.upper_pipe)),

            # Danger up
            (bird[1].colliderect(game.upper_pipe) or bird[1].colliderect(game.upper_pipe)),

            # Danger down
            (bird[2].colliderect(game.upper_pipe) or bird[2].colliderect(game.upper_pipe)),
            
            # Move direction
            dir_up,
            dir_down,
            
            # Next empty space location 
            game.space < bird_pos[1], #empty space above
            game.space > bird_pos[1] #empty space below
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
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
        self.epsilon = 80 - self.n_games
        final_move = [0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 1)
            final_move[0] = move
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            if(prediction >= 0.5):
                final_move[0] = 1
            else:
                final_move[0] = 0

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = FlappyBirdAI(500,400)
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.run(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()