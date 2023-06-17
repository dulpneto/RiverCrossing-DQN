import sys

import tensorflow as tf
import numpy as np

import cv2

from collections import deque
import timeit
import random

from environment.RiverCrossingEnv import RiverCrossingEnv
from agent import AgentModel as ag

from planning.ValueIteration import ValueIteration as ValueIteration

MIN_REPLAY_SIZE = 1000

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

QUIET = True

def main():
    # Building environment
    shape = (10, 10)
    #env = RiverCrossingEnv(shape, state_as_img=True)

    #print(env.reset())

    RiverCrossingEnv.draw_img_state(shape, 8)

    gamma = 0.99
    lamb = -1

    #policy, v, steps, updates, diffs, v_history = ValueIteration.run(env, lamb, gamma)

    h, w = shape
    for y in range(h):
        for x in range(w):
            s = x + (y * w)
            #RiverCrossingEnv.draw_img_state(shape, s)
    #process_state_image(0)

def process_state_image(state):
    image = cv2.imread('environment/img/river_{}.png'.format(state))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('environment/img/river_{}_2.png'.format(state), image_gray)
    image_gray = image_gray.astype(float)
    image_gray /= 255.0

    image = image.astype(float)
    image /= 255.0

    for i in range(len(image)):
        for j in range(len(image[i])):
            #for c in range(len(image[i][j])):
            print(image[i][j], image_gray[i][j])

    return image_gray


if __name__ == "__main__":
    main()



