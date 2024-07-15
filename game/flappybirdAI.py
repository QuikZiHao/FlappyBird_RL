import pygame
import sys
import random
import time
import numpy as np

class FlappyBirdAI:
    def __init__(self, window_size_width:int, window_size_height:int):
        self.window_size = (window_size_width,window_size_height)
        self.font_style = ("Jokerman Regular",50)
        self.clock = pygame.time.Clock()
        self.fps = 30
        self.speed = 1 #change this for training
        self.frame_iter = 0
        self.game_over = False
        self.bird_pos = (0,0)
        self.direction = 0
        self.bird = pygame.Rect(130, 150, 50, 50)
        self.upper_pipe = pygame.Rect(400, 0, 40, 120)
        self.lower_pipe = pygame.Rect(400, 280, 40, 120)
        self.space = 0

    def reset(self):
        self.frame_iter = self.frame_iter + 1
        self.run()
        
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

    def run(self, action):
        reward = 0
        pygame.init()
        mainWindow = pygame.display.set_mode((500, 400))
        bird_img_list = self.bird_img_init()
        background_image = pygame.image.load(r'game\source\background.png')
        background_image = pygame.transform.scale(background_image, (500, 400))
        upper_pipe_img =  pygame.image.load(r'game\source\pipe_top.png')
        upper_pipe_img = pygame.transform.scale(upper_pipe_img, (40, 120))
        bottom_pipe_img =  pygame.image.load(r'game\source\pipe_bottom.png')
        bottom_pipe_img = pygame.transform.scale(bottom_pipe_img, (40, 120))
        score = 0

        self.upper_pipe = pygame.Rect(400, 0, 40, 120)
        self.lower_pipe = pygame.Rect(400, 280, 40, 120)
        self.bird = pygame.Rect(130, 150, 50, 50)


        clock = pygame.time.Clock()
        speed_up_coeff = 0
        velocity_pipe = 5
        velocity_bird = 0
        accelration_bird = 1
        bird_idx = 0
        bird_ratio = 5

        my_font = pygame.font.SysFont("Jokerman Regular", 50)
        gameover_text = my_font.render("Game Over", True, (255, 0, 0))
        pause_text = my_font.render("Pause", True, (255, 0, 0))

        my_font = pygame.font.SysFont("Jokerman Regular", 30)
        play_again_text = my_font.render("Press 'SPACE' to play again or 'ESCAPE' to exit", True, (255, 0, 0))

        my_font = pygame.font.SysFont("Jokerman Regular", 35)
        score_text = my_font.render("Your Score: " + str(score), True, (255, 0, 0))
        score_renderer = my_font.render("0", True, (255, 0, 0))

        state = "playing"

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            if score % 30 == 0 and speed_up_coeff < 5 and score != 0:
                speed_up_coeff += 1

            mainWindow.blit(background_image, (0,0))
            mainWindow.blit(upper_pipe_img, (self.upper_pipe.x, self.upper_pipe.y))
            mainWindow.blit(bottom_pipe_img, (self.lower_pipe.x, self.lower_pipe.y))
            mainWindow.blit(bird_img_list[bird_idx // bird_ratio], (self.bird.x, self.bird.y))
            mainWindow.blit(score_renderer, (30, 30)) 

            if state == "beginning":
                keypressed = pygame.key.get_pressed()
                if keypressed[pygame.K_SPACE]:
                    state = "playing"
                    
            if state == "playing":
                self.game_over = False
                bird_idx += 1
                if bird_idx >= (bird_ratio * len(bird_img_list)):
                    bird_idx = 0

                self.upper_pipe.x -= velocity_pipe + speed_up_coeff
                self.lower_pipe.x -= velocity_pipe + speed_up_coeff

                velocity_bird += accelration_bird + speed_up_coeff
                self.bird.y += velocity_bird
                self.direction = 0 #down

                # keypressed = pygame.key.get_pressed()
                # if keypressed[pygame.K_SPACE]:
                #     velocity_bird = -7
                # if keypressed[pygame.K_p]:
                #     state = "pause"
                if np.array_equal(action, [1]):
                    velocity_bird = -7 #fly up
                    self.direction = 1 #up
                
                if self.upper_pipe.x <= -40:
                    self.upper_pipe.x = 500
                    self.lower_pipe.x = 500

                    self.upper_pipe.h = random.randint(40, 200)
                    upper_pipe_img = pygame.transform.scale(upper_pipe_img, (40, self.upper_pipe.h))
                    self.lower_pipe.h = 240 - self.upper_pipe.h
                    self.lower_pipe.y = 400 - self.lower_pipe.h
                    self.space = (self.upper_pipe.y + self.lower_pipe.y)/2
                    bottom_pipe_img = pygame.transform.scale(bottom_pipe_img, (40, self.lower_pipe.h))

                if self.bird.colliderect(self.upper_pipe) or self.bird.colliderect(self.lower_pipe) or self.bird.y < -30 or self.bird.y > 400:
                    state = "game over"
                    reward -= 10
                    self.game_over = True

                if self.bird.x == self.lower_pipe.x + self.lower_pipe.w :
                    score += 1
                    reward += 10
                    score_renderer = my_font.render(str(score), True, (255, 0, 0))

                self.bird_pos = (self.bird.x, self.bird.y)


            if state == "pause":
                mainWindow.blit(pause_text, (200, 150))

                keypressed = pygame.key.get_pressed()
                if keypressed[pygame.K_SPACE]:
                    state = "playing"

                
            if state == "game over":
                speed_up_coeff = 0
                
                mainWindow.blit(gameover_text, (160, 100))
                mainWindow.blit(play_again_text, (25, 300))

                score_text = my_font.render("Your Score: " + str(score), True, (255, 0, 0))
                mainWindow.blit(score_text, (180, 160))
                # time.sleep(2)

                keypressed = pygame.key.get_pressed()
                if keypressed[pygame.K_SPACE]:
                    self.bird.x = 130
                    self.bird.y = 150
                    self.upper_pipe.x = 400
                    self.lower_pipe.x = 400
                    score = 0
                    score_renderer = my_font.render(str(score), True, (255, 0, 0))
                    state = "playing"
                if keypressed[pygame.K_ESCAPE]:
                    pygame.quit()
                    sys.exit()

            self.clock.tick(self.speed*self.fps)
            pygame.display.update()

            return reward, self.game_over, score












