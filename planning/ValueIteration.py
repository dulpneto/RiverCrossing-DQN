
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
                        q[a] += t * (np.sign(lamb) * np.exp(lamb * (r + gamma * V[s_next])))

                    q_updates += 1

                V[s] = (np.log(np.sign(lamb) * max(q)) / lamb)

                policy[s] = np.argmax(q)
                updates += q_updates

            diffs.append(np.max(np.fabs(prev_V - V)))

            if np.max(np.fabs(prev_V - V)) < epsilon:
                break
        return policy, V, steps, updates, diffs, V_history