import tensorflow as tf
import numpy as np
import argparse

import os

from collections import deque
import timeit
import random
import math

from environment.RiverCrossingEnv import RiverCrossingEnv
from agent import AgentModel as ag

MIN_REPLAY_SIZE = 1000

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# minimum rewards allowed while training
# this is done to avoid being stuck
MIN_TRAIN_REWARDS = -5000

#RUN_FOR = 'ql2'
RUN_FOR = 'dqn2'

def log(txt, type):
    with open('./logs/{}/result_{}_2.log'.format(RUN_FOR, type), 'a') as f:
        f.write(txt + '\n')
    print(txt)



def run(bellman_update):

    if not os.path.exists('logs/{}'.format(RUN_FOR)):
        os.makedirs('logs/{}'.format(RUN_FOR))

    alpha = 0.1
    gamma = 0.99

    epsilon = 0.3  # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start

    # An episode a full game
    train_episodes = 150

    shape = (10, 10)

    # Building environment
    env = RiverCrossingEnv(shape, state_as_img=False)
    if RUN_FOR == 'ql2':
        agent_type = 'QL'
    else:
        agent_type = 'DQN_CACHED'

    h, w = env.shape

    #for lamb in [-1.5, -1.0, -0.75, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 0.75, 1.0, 1.5]:
    for lamb in [-0.75, -0.5, -0.2, -0.1, 0.1, 0.2, 0.5, 0.75, 1.0, 1.5]:
    #for lamb in [-1.5, -0.5, 0.5, 1.5]:
    #for l in range(-15, 16, 1):
        #lamb = l / 10

        if lamb == 0:
            print('skip 0')
            continue

        policy_stack = {}

        for y in range(h):
            for x in range(w):
                s = x + (y * w)
                policy_stack[s] = []

        samples = 25
        for sample in range(1, samples+1):

            # 1. Initialize the Target and Main models
            # Main Model (updated every step)
            model = ag.AgentModel.build(agent_type, bellman_update, env, alpha, gamma, lamb)
            model.load_model()

            # Target Model (updated every 100 steps)
            target_model = ag.AgentModel.build(agent_type, bellman_update, env, alpha, gamma, lamb)
            target_model.set_weights(model)

            replay_memory = deque(maxlen=1_000)

            steps_to_update_target_model = 0

            error=False

            for episode in range(1, train_episodes+1):

                if error:
                    break

                total_training_rewards = 0
                state = env.reset()
                done = False
                while not done:
                    steps_to_update_target_model += 1

                    # 2. Explore using the Epsilon Greedy Exploration Strategy
                    random_number = np.random.rand()
                    if random_number <= epsilon:
                        # Explore
                        action = env.action_space.sample()
                    else:
                        # Exploit best known action
                        predicted = model.find_qs(state)
                        action = np.argmax(predicted)

                    # step
                    if total_training_rewards < MIN_TRAIN_REWARDS:
                        new_state, reward, done, info = env.step_safe()
                    else:
                        new_state, reward, done, info = env.step(action)

                    # keeping replay memory to batch training
                    replay_memory.append([state, action, reward, new_state, done])

                    # 3. Update the Main Network using the Bellman Equation
                    if steps_to_update_target_model % 4 == 0 or done:
                        try:
                            train(replay_memory, model, target_model)
                        except (ValueError, OverflowError):
                            log('{}\t{}\t{}\t{}'.format(lamb, bellman_update, episode, 1), 'error')
                            error = True
                            break

                    state = new_state
                    total_training_rewards += reward

                    if done:
                        predicted = model.find_qs(env.s0)
                        vs0 = max(predicted)

                        if math.isnan(vs0):
                            log('{}\t{}\t{}\t{}'.format(lamb, bellman_update, episode, 1), 'error')
                            error = True
                            break

                        log('{}\t{}\t{}\t{}'.format(lamb, bellman_update, episode, vs0), 'vso')
                        log('{}\t{}\t{}\t{}'.format(lamb, bellman_update, episode, model.bellman.max_u), 'us')
                        log('{}\t{}\t{}\t{}'.format(lamb, bellman_update, episode, total_training_rewards), 'rewards')
                        model.bellman.max_u = 0

                        target_model.set_weights(model)
                        steps_to_update_target_model = 0
                        break

def train(replay_memory, model, target_model):

    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = 100
    mini_batch = random.sample(replay_memory, batch_size)

    model.update_model(mini_batch, target_model)


def main():
    try:
        # record start time
        t_0 = timeit.default_timer()

        # running
        #run('Target')
        run('TD')

        # calculate elapsed time and print
        t_1 = timeit.default_timer()
        elapsed_time = round((t_1 - t_0), 3)
        print(f"Elapsed time: {elapsed_time} s")
    except (ValueError, OverflowError) as error:
        print(error)
        print('Error')


if __name__ == "__main__":
    main()



