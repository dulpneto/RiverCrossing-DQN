
import numpy as np


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