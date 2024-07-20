from game.flappybird import FlappyBird
from game.flappybirdAI import FlappyBirdAI, Train
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",type=str,default='train')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if(args.mode == 'play'):
        game = FlappyBird(500,400)
        # game = FlappyBirdAI(500,400)
        game.run()

    elif(args.mode == 'train'):
        game = Train()
        game.train()
