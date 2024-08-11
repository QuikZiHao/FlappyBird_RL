import pygame
import sys
import random
import time
import numpy as np
from agent import Agent
from model.helper import plot

pygame.init()

class FlappyBirdAI:
    def __init__(self, window_size_width:int, window_size_height:int, train_speed):
        self.first_start = True
        self.window_size = (window_size_width,window_size_height)
        # Make whole screen as 1 in state input
        self.state_arr = np.ones((self.window_size[1], self.window_size[0]))
        
        self.font_style = ("Jokerman Regular",50)
        self.clock = pygame.time.Clock()
        self.fps = 30
        self.speed = train_speed #change this for training
        self.frame_iter = 0
        self.game_over = False
        self.background_image, self.upper_pipe_img, self.bottom_pipe_img = self.game_init()
        self.direction = 0
        self.score = 0
        self.reward = 0 
        self.speed_up_coeff = 0
        self.velocity_pipe = 5
        self.velocity_bird = 0
        self.accelration_bird = 1
        self.bird_idx = 0
        self.bird_ratio = 5
        self.top_bot_limit_penalty = -200
        self.pass_reward = 30

        self.main_window = pygame.display.set_mode((500, 400))
        self.bird_img_list = self.bird_img_init()
        self.upper_pipe = pygame.Rect(400, 0, 40, 120)
        self.lower_pipe = pygame.Rect(400, 280, 40, 120)
        self.bird = pygame.Rect(130, 150, 50, 50)
        self.clock = pygame.time.Clock()
        self.my_font = pygame.font.SysFont("Jokerman Regular", 50)
        self.gameover_text = self.my_font.render("Game Over", True, (255, 0, 0))
        self.pause_text = self.my_font.render("Pause", True, (255, 0, 0))
        self.my_font = pygame.font.SysFont("Jokerman Regular", 30)
        self.play_again_text = self.my_font.render("Press 'SPACE' to play again or 'ESCAPE' to exit", True, (255, 0, 0))
        self.my_font = pygame.font.SysFont("Jokerman Regular", 35)
        self.score_text = self.my_font.render("Your Score: " + str(self.score), True, (255, 0, 0))
        self.score_renderer = self.my_font.render("0", True, (255, 0, 0))

        for i in range(self.upper_pipe.w):
            penalty = [0,0]
            for j in range(self.upper_pipe.h - 1,-1,-1):
                self.state_arr[j,self.upper_pipe.x + i] = -10 - penalty[0]
                penalty[0] += 1
            for k in range(self.lower_pipe.h):
                self.state_arr[self.lower_pipe.y + k,self.lower_pipe.x + i] = -10 - penalty[1]
                penalty[1] += 1
            for l in range(self.upper_pipe.h, self.lower_pipe.y):
                self.state_arr[l, self.upper_pipe.x + i] = self.pass_reward
        # Make top and bottom screen into self.top_bot_limit_penalty
        self.state_arr[0] = self.top_bot_limit_penalty
        self.state_arr[self.window_size[1]-1] = self.top_bot_limit_penalty
    def state_arr_init(self):
        # Make whole screen as 1 in state input
        self.state_arr = np.ones((self.window_size[1], self.window_size[0]))
        print(self.upper_pipe.x)
        # Find the pipe position
        for i in range(self.upper_pipe.w):
            penalty = [0,0]
            for j in range(self.upper_pipe.h - 1,-1,-1):
                self.state_arr[j,self.upper_pipe.x + i] = -10 - penalty[0]
                penalty[0] += 1
            for k in range(self.lower_pipe.h):
                self.state_arr[self.lower_pipe.y + k,self.lower_pipe.x + i] = -10 - penalty[1]
                penalty[1] += 1
            for l in range(self.upper_pipe.h, self.lower_pipe.y):
                self.state_arr[l, self.upper_pipe.x + i] = self.pass_reward
        # Make top and bottom screen into self.top_bot_limit_penalty
        self.state_arr[0] = self.top_bot_limit_penalty
        self.state_arr[self.window_size[1]-1] = self.top_bot_limit_penalty

    def pop_state_arr(self):
        self.state_arr = np.delete(self.state_arr, np.s_[:5], axis=1)

    def append_state_arr(self, new_column:np.array):
        self.state_arr = np.hstack((self.state_arr, new_column))

    def reset(self):
        self.frame_iter = self.frame_iter + 1
        self.upper_pipe = pygame.Rect(400, 0, 40, 120)
        self.lower_pipe = pygame.Rect(400, 280, 40, 120)
        self.bird = pygame.Rect(130, 150, 50, 50)
        self.reward = 0
        self.score = 0
        self.speed_up_coeff = 0
        self.velocity_pipe = 5
        self.velocity_bird = 0
        self.accelration_bird = 1
        self.bird_idx = 0
        self.bird_ratio = 5
        self.state_arr_init()
        
    def bird_img_init(self):
        bird1_image = pygame.image.load(r'game\source\bird1.png')
        bird1_image = pygame.transform.scale(bird1_image, (50, 50))
        bird2_image = pygame.image.load(r'game\source\bird2.png')
        bird2_image = pygame.transform.scale(bird2_image, (50, 50))
        bird3_image = pygame.image.load(r'game\source\bird3.png')
        bird3_image = pygame.transform.scale(bird3_image, (50, 50))
        bird4_image = pygame.image.load(r'game\source\bird4.png')
        bird4_image = pygame.transform.scale(bird4_image, (50, 50))
        return [bird1_image, bird2_image, bird3_image, bird4_image]
    
    def game_init(self):
        background_image = pygame.image.load(r'game\source\background.png')
        background_image = pygame.transform.scale(background_image, (500, 400))
        upper_pipe_img =  pygame.image.load(r'game\source\pipe_top.png')
        upper_pipe_img = pygame.transform.scale(upper_pipe_img, (40, 120))
        bottom_pipe_img =  pygame.image.load(r'game\source\pipe_bottom.png')
        bottom_pipe_img = pygame.transform.scale(bottom_pipe_img, (40, 120))
        return background_image, upper_pipe_img, bottom_pipe_img

    def _run(self, action):
        if self.first_start:
            self.state_arr_init()
            self.first_start = False

        state = "playing"

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if self.score % 30 == 0 and self.speed_up_coeff < 5 and self.score != 0:
            if speed_up_flag == True:
                speed_up_flag = True
                self.speed_up_coeff += 1

        self.main_window.blit(self.background_image, (0,0))
        self.main_window.blit(self.upper_pipe_img, (self.upper_pipe.x, self.upper_pipe.y))
        self.main_window.blit(self.bottom_pipe_img, (self.lower_pipe.x, self.lower_pipe.y))
        self.main_window.blit(self.bird_img_list[self.bird_idx // self.bird_ratio], (self.bird.x, self.bird.y))
        self.main_window.blit(self.score_renderer, (30, 30)) 

        if state == "playing":
            if np.array_equal(action, [1]):
                self.velocity_bird = -7 #fly up
                self.direction = 1 #up
            else:
                self.direction = 0 #down
            self.game_over = False
            self.bird_idx += 1
            if self.bird_idx >= (self.bird_ratio * len(self.bird_img_list)):
                self.bird_idx = 0

            self.upper_pipe.x -= self.velocity_pipe + self.speed_up_coeff
            self.lower_pipe.x -= self.velocity_pipe + self.speed_up_coeff

            self.velocity_bird += self.accelration_bird + self.speed_up_coeff
            self.bird.y += self.velocity_bird
            
            if self.upper_pipe.x <= -40:
                self.upper_pipe.x = 500
                self.lower_pipe.x = 500

                self.upper_pipe.h = random.randint(40, 200)
                self.upper_pipe_img = pygame.transform.scale(self.upper_pipe_img, (40, self.upper_pipe.h))
                self.lower_pipe.h = 240 - self.upper_pipe.h
                self.lower_pipe.y = 400 - self.lower_pipe.h

                self.bottom_pipe_img = pygame.transform.scale(self.bottom_pipe_img, (40, self.lower_pipe.h))

            self.pop_state_arr()
            new_col = np.ones([self.window_size[1],self.velocity_pipe])
            # Check if new pipe is generated
            if self.upper_pipe.x <= 500 and self.upper_pipe.x > 500 - self.upper_pipe.w:
                # Make a new column everytime a new frame generated
                penalty = [0,0]
                for j in range(self.upper_pipe.h - 1,-1,-1):
                    new_col[j,:] = -10 - penalty[0]
                    penalty[0] += 1
                for k in range(self.lower_pipe.h):
                    new_col[self.lower_pipe.y + k,:] = -10 - penalty[1]
                    penalty[1] += 1
                for l in range(self.upper_pipe.h, self.lower_pipe.y):
                    new_col[l,:] = self.pass_reward

            new_col[0,:] = self.top_bot_limit_penalty
            new_col[self.window_size[1] - 1,:] = self.top_bot_limit_penalty
            self.append_state_arr(new_column=new_col)
            if(self.bird.y <= 0):
                self.reward += self.state_arr[0, self.bird.x + self.bird.width]
            elif(self.bird.y >= 400):
                self.reward += self.state_arr[0, self.bird.x + self.bird.width]
            elif(self.bird.y > 0 and self.bird.y < 400):
                self.reward += self.state_arr[self.bird.y, self.bird.x + self.bird.width]

            if self.bird.colliderect(self.upper_pipe) or self.bird.colliderect(self.lower_pipe) or self.bird.y <= 0 or self.bird.y > 400:
                state = "game over"
                self.game_over = True

            if self.bird.x == self.lower_pipe.x + self.lower_pipe.w :
                self.score += 1
                self.reward += 350
                self.score_renderer = self.my_font.render(str(self.score), True, (255, 0, 0))

        if state == "pause":
            self.main_window.blit(self.pause_text, (200, 150))

            keypressed = pygame.key.get_pressed()
            if keypressed[pygame.K_SPACE]:
                state = "playing"
            
        if state == "game over":
            self.speed_up_coeff = 0
            
            self.main_window.blit(self.gameover_text, (160, 100))
            self.main_window.blit(self.play_again_text, (25, 300))

            score_text = self.my_font.render("Your Score: " + str(self.score), True, (255, 0, 0))
            self.main_window.blit(score_text, (180, 160))

        self.clock.tick(self.speed*self.fps)
        pygame.display.update()

        return self.reward/10, self.game_over, self.score


class Train:
    def __init__(self, train_speed = 1):
        self.plot_scores = []
        self.plot_mean_scores = []
        self.total_score = 0
        self.record = 0
        self.train_speed = train_speed
        self.agent = Agent()
        self.game = FlappyBirdAI(500, 400, self.train_speed)

    def train(self):
        while True:
            # get old state
            state_old = self.game.state_arr

            # get move
            final_move = self.agent.get_action(state_old)

            # perform move and get new state
            reward, done, score = self.game._run(final_move)
            self.state_new = self.agent.get_state(self.game)

            # train short memory
            self.agent.train_short_memory(state_old, final_move, reward, self.state_new, done)

            # remember
            self.agent.remember(state_old, final_move, reward, self.state_new, done)

            if done:
                # train long memory, plot result
                self.game.reset()
                self.agent.n_games += 1
                self.agent.train_long_memory()

                if score > self.record:
                    self.record = score
                    self.agent.model.save()

                print('Game', self.agent.n_games, 'Score', score, 'Record:', self.record, 'Reward:', reward)

                self.plot_scores.append(score)
                self.total_score += score
                mean_score = self.total_score / self.agent.n_games
                self.plot_mean_scores.append(mean_score)
                plot(self.plot_scores, self.plot_mean_scores)
