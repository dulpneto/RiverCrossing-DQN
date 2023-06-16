import sys

import tensorflow as tf
import numpy as np

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
    shape = (5, 4)
    #env = RiverCrossingEnv(shape)

    gamma = 0.99
    lamb = -1

    #policy, v, steps, updates, diffs, v_history = ValueIteration.run(env, lamb, gamma)

    h, w = shape
    for y in range(h):
        for x in range(w):
            s = x + (y * w)
            RiverCrossingEnv.draw_img_state(shape, s)


if __name__ == "__main__":
    main()



