from game.flappybird import FlappyBird
from game.flappybirdAI import FlappyBirdAI


if __name__ == "__main__":
    game = FlappyBird(500,400)
    # game = FlappyBirdAI(500,400)
    game.run()