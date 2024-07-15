import pygame

class FlappyBird:
    def __init__(self, window_size_width:int, window_size_height:int, pipe_distance:int):
        self.window_size = (window_size_width,window_size_height)
        self.pipe_distance = pipe_distance
        self.font_style = ("Jokerman Regular",50)


    def bird_img_init():
        bird1_image = pygame.image.load(r'source\bird1.png')
        bird1_image = pygame.transform.scale(bird1_image, (50, 50))
        bird2_image = pygame.image.load(r'source\bird2.png')
        bird2_image = pygame.transform.scale(bird2_image, (50, 50))
        bird3_image = pygame.image.load(r'source\bird3.png')
        bird3_image = pygame.transform.scale(bird3_image, (50, 50))
        bird4_image = pygame.image.load(r'source\bird4.png')
        bird4_image = pygame.transform.scale(bird4_image, (50, 50))
        return ["bird_normal_",bird1_image, bird2_image, bird3_image, bird4_image]

    def run(self):
        pygame.init()
        mainWindow = pygame.display.set_mode((500, 400))

game = FlappyBird.run()

