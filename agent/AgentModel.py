from abc import ABC, abstractmethod

from agent import BellmanUpdate as bu

import tensorflow as tf
import numpy as np
from tensorflow import keras
from copy import copy
from collections import defaultdict

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Types of agents implemented
class AgentType:
    QL_TARGET = 'QL_TARGET'
    QL_TD = 'QL_TD'
    QL_TARGET_LSE = 'QL_TARGET_LSE'
    DQN_TARGET = 'DQN_TARGET'
    DQN_TD = 'DQN_TD'
    DQN_TARGET_LSE = 'DQN_TARGET_LSE'
    DQN_TARGET_CACHE = 'DQN_TARGET_CACHE'
    DQN_TD_CACHE = 'DQN_TD_CACHE'
    DQN_TARGET_LSE_CACHE = 'DQN_TARGET_LSE_CACHE'


# Agent default behaviour
class AgentModel(ABC):

    def __init__(self, env, alpha, gamma, lamb, bellman_type):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.lamb = lamb
        self.bellman_type = bellman_type
        self.bellman = bu.BellmanUpdate.build(bellman_type, alpha, gamma, lamb)
        self.create_model()

    def reset(self, alpha, gamma, lamb, bellman_type):
        self.alpha = alpha
        self.gamma = gamma
        self.lamb = lamb
        self.bellman = bu.BellmanUpdate.build(bellman_type, alpha, gamma, lamb)

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def update_model(self, mini_batch, target_model):
        pass

    @abstractmethod
    def set_weights(self, model):
        pass

    @abstractmethod
    def find_qs(self, state):
        pass

    @abstractmethod
    def print_qs_model(self):
        pass

    @staticmethod
    def build(type, env, alpha, gamma, lamb):
        if type == AgentType.QL_TARGET:
            return QLearningModel(env, alpha, gamma, lamb, bu.Type.TARGET)
        elif type == AgentType.QL_TD:
            return QLearningModel(env, alpha, gamma, lamb, bu.Type.TD)
        elif type == AgentType.QL_TARGET_LSE:
            return QLearningModel(env, alpha, gamma, lamb, bu.Type.TARGET_LOG_SUM_EXP)
        elif type == AgentType.DQN_TARGET:
            return DQNModel(env, alpha, gamma, lamb, bu.Type.TARGET)
        elif type == AgentType.DQN_TD:
            return DQNModel(env, alpha, gamma, lamb, bu.Type.TD)
        elif type == AgentType.DQN_TARGET_LSE:
            return DQNModel(env, alpha, gamma, lamb, bu.Type.TARGET_LOG_SUM_EXP)
        elif type == AgentType.DQN_TARGET_CACHE:
            return DQNModelCached(env, alpha, gamma, lamb, bu.Type.TARGET)
        elif type == AgentType.DQN_TD_CACHE:
            return DQNModelCached(env, alpha, gamma, lamb, bu.Type.TD)
        elif type == AgentType.DQN_TARGET_LSE_CACHE:
            return DQNModelCached(env, alpha, gamma, lamb, bu.Type.TARGET_LOG_SUM_EXP)
        else:
            raise Exception("Not implemented {}".format(type))
        return


# QLearning
class QLearningModel(AgentModel):

    def create_model(self):
        self.Q = defaultdict(lambda: np.ones(self.env.action_space.n) * self.bellman.default_V)

    def load_model(self):
        # do nothing
        return

    def update_model(self, mini_batch, target_model):
        for index, (current_state, action, reward, next_state, done) in enumerate(mini_batch):
            future_qs = target_model.Q[next_state]
            current_qs = self.Q[current_state]
            self.Q[current_state][action] = self.bellman.bellman_update(current_qs, future_qs, action, reward)

    def set_weights(self, model):
        # deep copy Q
        self.Q = copy(model.Q)

    def find_qs(self, state):
        # Q[s]
        return self.Q[state]

    def print_qs_model(self):
        pass


# DQN with original implementation without cache strategy
class DQNModel(AgentModel):

    def create_model(self):
        state_shape = self.env.observation_space.shape
        action_shape = self.env.action_space.n

        init = tf.keras.initializers.HeUniform()
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(24, input_shape=state_shape, activation='relu', kernel_initializer=init))
        self.model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
        self.model.add(keras.layers.Dense(action_shape, activation='linear', kernel_initializer=init))
        self.model.compile(loss=tf.keras.losses.Huber(),
                           optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])

        self.cached_q = False

    def load_model(self):
        # treinar ou nao?
        return

    def update_model(self, mini_batch, target_model):

        current_states = np.array(
            [self.encode_observation(transition[0]) for transition in mini_batch])
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array(
            [self.encode_observation(transition[3]) for transition in mini_batch])
        future_qs_list = target_model.model.predict(new_current_states)

        x = []
        y = []
        for index, (current_state, action, reward, next_state, done) in enumerate(mini_batch):
            future_qs = future_qs_list[index]
            current_qs = current_qs_list[index]

            current_qs[action] = self.bellman.bellman_update(current_qs, future_qs, action, reward)

            x.append(self.encode_observation(current_state))
            y.append(current_qs)

        self.model.fit(np.array(x), np.array(y), batch_size=len(mini_batch), verbose=0, shuffle=True)

    def set_weights(self, model):
        self.model.set_weights(model.model.get_weights())

    def find_qs(self, state):
        encoded = self.encode_observation(state)
        encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
        predicted = self.model.predict(encoded_reshaped).flatten()

        return predicted

    def print_qs_model(self):
        pass

    def encode_observation(self, state):
        encoded = np.zeros(self.env.observation_space.shape)
        encoded[int(state)] = 1
        return encoded


# DQN with Cache strategy to make testes go faster
class DQNModelCached(DQNModel):

    # when false skip using model - use only Q table but model is trained
    MODEL_USAGE = True

    def update_model(self, mini_batch, target_model):

        for index, (current_state, action, reward, next_state, done) in enumerate(mini_batch):
            future_qs = target_model.find_qs(next_state)
            current_qs = self.find_qs(current_state)
            self.Q[current_state][action] = self.bellman.bellman_update(current_qs, future_qs, action, reward)

        x = [self.encode_observation(s) for s in range(self.env.observation_space.shape[0])]
        y = [self.bellman.bellman_normalize(self.Q[s]) for s in range(self.env.observation_space.shape[0])]

        self.model.fit(np.array(x), np.array(y), batch_size=len(mini_batch), verbose=0, shuffle=True)

        # marking cache to be refreshed
        # self.cached_q = False

    def set_weights(self, model):
        # WITH MODEL USAGE
        if self.MODEL_USAGE:
            self.cached_q = False
            self.model.set_weights(model.model.get_weights())
        else:  # SKIP MODEL USAGE
            if not model.cached_q:
                self.Q = defaultdict(lambda: np.ones(self.env.action_space.n) * self.bellman.default_V)
            else:
                self.Q = model.Q

    def find_qs(self, state):

        # making a batch prediction to speed up tests
        if not self.cached_q:
            # WITH MODEL USAGE
            if self.MODEL_USAGE:
                x = [self.encode_observation(s) for s in range(self.env.observation_space.shape[0])]
                prediction = self.model.predict(np.array(x), batch_size=len(x))
                self.Q = [self.bellman.bellman_denormalize(qs) for qs in prediction]
            else:  # SKIP MODEL USAGE
                self.Q = defaultdict(lambda: np.ones(self.env.action_space.n) * self.bellman.default_V)

            self.cached_q = True

        return self.Q[state]

    def print_qs_model(self):

        h, w = self.env.shape
        lineState = ''
        for y in range(h):
            for x in range(w):
                state = x + (y * w)
                encoded = self.encode_observation(state)
                encoded_reshaped = encoded.reshape([1, encoded.shape[0]])
                q_s = self.model.predict(encoded_reshaped).flatten()
                lineState = '{}\t{}'.format(lineState, str(round(max(q_s), 3)))
            lineState = '{}\n'.format(lineState)
        print('')
        print(lineState)
