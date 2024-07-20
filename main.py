from game.flappybird import FlappyBird
from game.flappybirdAI import FlappyBirdAI
from agent import train
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",type=str,default='play')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if(args.mode == 'play'):
        game = FlappyBird(500,400)
        # game = FlappyBirdAI(500,400)
        game.run()

    elif(args.mode == 'train'):
        train()
