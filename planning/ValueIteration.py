
import numpy as np

from collections import defaultdict
from copy import deepcopy


class ValueIteration:

    @staticmethod
    def run(env, lamb, gamma, epsilon=1e-3):
        V = np.zeros(env.observation_space.shape[0])
        policy = np.zeros(env.observation_space.shape[0])
        steps = 0
        updates = 0
        diffs = []
        V_history = []
        while True:
            steps += 1
            prev_V = np.copy(V)
            V_history.append(np.copy(V))
            for s in range(env.observation_space.shape[0]):
                # calculating action value
                q = np.zeros(env.action_space.n)
                q_updates = 0
                for a in range(env.action_space.n):
                    for s_next, t, r in env.P[(s, a)]:
                        if lamb == 0:
                            q[a] += t * (r + gamma * V[s_next])
                        else:
                            q[a] += t * (np.sign(lamb) * np.exp(lamb * (r + gamma * V[s_next])))

                    q_updates += 1

                if lamb == 0:
                    V[s] = max(q)
                else:
                    V[s] = (np.log(np.sign(lamb) * max(q)) / lamb)

                policy[s] = np.argmax(q)
                updates += q_updates

            diffs.append(np.max(np.fabs(prev_V - V)))

            if np.max(np.fabs(prev_V - V)) < epsilon:
                break
        return policy, V, steps, updates, diffs, V_history

    @staticmethod
    def run_target(env, lamb, gamma, alpha, epsilon=1e-3):

        Vu = np.zeros(env.observation_space.shape[0])
        # utility of zero
        V = np.ones(env.observation_space.shape[0]) * np.sign(lamb)
        Q = defaultdict(lambda: np.ones(env.action_space.n) * np.sign(lamb))

        policy = np.zeros(env.observation_space.shape[0])
        steps = 0
        updates = 0
        diffs = []
        V_history = []

        utilities_per_step = []

        while True:
            steps += 1

            # storing on utility
            prev_V = np.copy(Vu)
            V_history.append(np.copy(Vu))

            prev_Q = deepcopy(Q)
            utilities = []
            for s in range(env.observation_space.shape[0]):
                # calculating action value
                q_updates = 0
                for a in range(env.action_space.n):
                    q_a = 0
                    for s_next, t, r in env.P[(s, a)]:
                        if lamb == 0:
                            q_a += t * (r + gamma * V[s_next])
                        else:
                            q_r = (np.log(np.sign(lamb) * np.max(prev_Q[s_next])) / lamb)
                            target = r + (gamma * q_r)
                            u = (np.sign(lamb) * np.exp(lamb * target))
                            #if s == env.s0:
                            utilities.append(u)
                            q_a += t * (u - prev_Q[s][a])

                    Q[s][a] = prev_Q[s][a] + (alpha * (q_a))
                    q_updates += 1

                policy[s] = np.argmax(Q[s])
                V[s] = np.max(Q[s])

                updates += q_updates

            # putting on utility to compare
            Vu = (np.log(np.sign(lamb) * V) / lamb)
            diffs.append(np.max(np.fabs(prev_V - Vu)))

            if lamb > 0:
                utilities_per_step.append(max(utilities))
            else:
                utilities_per_step.append(min(utilities))

            if np.max(np.fabs(prev_V - Vu)) < epsilon:
                break
        return policy, Vu, steps, updates, diffs, V_history, utilities_per_step

    @staticmethod
    def run_td(env, lamb, gamma, alpha, epsilon=1e-3):
        V = np.zeros(env.observation_space.shape[0])
        Q = defaultdict(lambda: np.zeros(env.action_space.n))
        policy = np.zeros(env.observation_space.shape[0])
        steps = 0
        updates = 0
        diffs = []
        V_history = []
        y_0 = 0
        x_0 = np.sign(lamb) * np.exp(lamb * y_0)

        utilities_per_step = []

        while True:
            steps += 1
            prev_V = np.copy(V)
            V_history.append(np.copy(V))
            prev_Q = deepcopy(Q)

            utilities = []
            for s in range(env.observation_space.shape[0]):
                # calculating action value
                q_updates = 0
                for a in range(env.action_space.n):
                    q_a = 0
                    for s_next, t, r in env.P[(s, a)]:
                        if lamb == 0:
                            q_a += t * (r + gamma * V[s_next])
                        else:
                            td = r + (gamma * np.max(prev_Q[s_next])) - prev_Q[s][a]
                            u = np.sign(lamb) * np.exp(lamb * td)
                            #if s == env.s0:
                            utilities.append(u)
                            q_a += t * u

                    Q[s][a] = prev_Q[s][a] + (alpha * (q_a - x_0))
                    q_updates += 1

                policy[s] = np.argmax(Q[s])
                V[s] = np.max(Q[s])
                updates += q_updates

            diffs.append(np.max(np.fabs(prev_V - V)))

            if lamb > 0:
                utilities_per_step.append(max(utilities))
            else:
                utilities_per_step.append(min(utilities))

            # print('\r','Iteration {}'.format(steps, round(np.max(np.fabs(prev_V - V)),3)), end='')

            if np.max(np.fabs(prev_V - V)) < epsilon:
                break
        return policy, V, steps, updates, diffs, V_history, utilities_per_step


    @staticmethod
    def find_safe_points(env, policy):
        count = 0
        h, w = env.shape
        for state in range((h * w) - 1, -1, -1):
            x = int(state % w)

            if x == 0:
                if policy[state] == 0:
                    count += 1
                else:
                    return count
        return count


