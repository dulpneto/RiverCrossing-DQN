import tensorflow as tf
import numpy as np

import argparse

from collections import deque
import timeit
import random

from environment.RiverCrossingEnv import RiverCrossingEnv
from agent import AgentModel as ag

MIN_REPLAY_SIZE = 1000

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

QUIET = False


def run():

    parser = argparse.ArgumentParser(description='Run DQN for River Crossing domain.')
    parser.add_argument('-t', '--type', type=float, default=0.0, help='The rtype of algorithm, default QL_TARGET_LSE.')
    parser.add_argument('-l', '--lamb', type=float, default=0.0, help='The risk param, default to 0.0.')
    parser.add_argument('-g', '--gamma', type=float, default=0.99, help='The discount factor, default to 0.99.')
    parser.add_argument('-a', '--alpha', type=float, default=0.1, help='The learning rate, default to 0.1.')
    parser.add_argument('-p', '--epsilon', type=float, default=0.3, help='The epsilon, default to 0.3.')
    parser.add_argument('-e', '--episodes', type=int, default=150, help='The numer of episodes, default to 150.')
    parser.add_argument('-sh', '--shape_h', type=int, default=5, help='The shape h size, default to 5.')
    parser.add_argument('-sw', '--shape_w', type=int, default=4, help='he shape w size, default to 4.')
    args = parser.parse_args()

    agent_type = ag.AgentType.QL_TARGET_LSE
    alpha = args.alpha
    gamma = args.gamma
    lamb = args.lamb

    epsilon = args.epsilon  # Epsilon-greedy algorithm in initialized at 1 meaning every step is random at the start

    # An episode a full game
    train_episodes = args.episodes

    # Building environment
    shape = (args.shape_h, args.shape_w)
    env = RiverCrossingEnv(shape)

    # 1. Initialize the Target and Main models
    # Main Model (updated every step)
    model = ag.AgentModel.build(agent_type, env, alpha, gamma, lamb)
    model.load_model()

    # Target Model (updated every 100 steps)
    target_model = ag.AgentModel.build(agent_type, env, alpha, gamma, lamb)
    target_model.set_weights(model)

    replay_memory = deque(maxlen=1_000)

    steps_to_update_target_model = 0

    for episode in range(train_episodes):
        if not QUIET:
            print('Episode: {}'.format(episode))

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
            new_state, reward, done, info = env.step(action)

            # keeping replay memory to batch training
            replay_memory.append([state, action, reward, new_state, done])

            # 3. Update the Main Network using the Bellman Equation
            if steps_to_update_target_model % 4 == 0 or done:
                train(replay_memory, model, target_model)

            state = new_state
            total_training_rewards += reward

            if done:
                if not QUIET:
                    print('Total training rewards: {} after n steps = {} with final reward = {}'.format(
                        total_training_rewards, episode, reward))
                total_training_rewards += 1
                if not QUIET:
                    env.render(model)
                    model.print_qs_model()
                    print('Copying main network weights to the target network weights')
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
        run()

        # calculate elapsed time and print
        t_1 = timeit.default_timer()
        elapsed_time = round((t_1 - t_0), 3)
        if not QUIET:
            print(f"Elapsed time: {elapsed_time} s")
    except (ValueError, OverflowError) as error:
        print(error)
        print('Error')


if __name__ == "__main__":
    main()



