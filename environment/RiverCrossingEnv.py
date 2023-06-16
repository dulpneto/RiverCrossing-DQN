import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2

# number of action - North, South, East, West
N_DISCRETE_ACTIONS = 4

STATE_IMG_WIDTH = 500


class RiverCrossingEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, shape, state_as_img=False, s0=None):
        # super(RiverCrossingEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

        self.shape = shape
        h, w = shape

        # self.observation_space = spaces.Discrete(w * h)
        self.observation_space = spaces.Box(low=0, high=1, shape=(w * h,))

        # when setting state as ima env return a plot of the state instead of the state number
        self.state_as_img = state_as_img
        # creating a cache to keep formatted images. This will make faster to run
        self.state_img_cache = {}
        self.state_img_width = STATE_IMG_WIDTH

        # probabilities - (state,action)[(state_next, probability,reward),...]
        self.P = {}

        # terminals
        self.G = []
        self.G.append(h * w - 1)
        # self.G.append(15)

        self.maxR = 1

        # defines s0 as the left bottom corner
        if s0 is None:
            self.s0 = w * (h - 1)
        else:
            self.s0 = s0

        self.current_s = self.s0

        for y in range(h):
            for x in range(w):
                s = x + (y * w)
                if self.state_as_img:
                    img = RiverCrossingEnv.draw_img_state(shape, s)
                    state_img = RiverCrossingEnv.process_state_image(img)
                    self.state_img_cache[s] = state_img

                for a in range(N_DISCRETE_ACTIONS):

                    s_next = RiverCrossingEnv.find_s_next(x, y, a, shape)

                    default_reward = -1

                    reward = default_reward
                    if (s_next in self.G):
                        reward = 0

                    if s == s_next:
                        default_reward = default_reward
                    if x == w - 1 and y == h - 1:  # meta
                        self.P[(s, a)] = [(s, 1, 0)]
                    # elif x == w - 1 and y == h - 2:  # meta 2
                    #    self.P[(s, a)] = [(s, 1, 0)]
                    elif x == 0:  # margem direita
                        self.P[(s, a)] = [(s_next, 1, reward)]
                    elif x == w - 1:  # margem esquerda
                        self.P[(s, a)] = [(s_next, 1, reward)]
                    elif y == 0:  # ponte
                        self.P[(s, a)] = [(s_next, 1, reward)]
                    elif y == h - 1:  # cachoeira
                        # P[(s, a)] = [(s, 1, -1)] #always stuck
                        self.P[(s, a)] = [(self.s0, 1, default_reward)]  # always returns to s0
                        # P[(s, a)] = [(0, 0.999, default_reward),(s, 0.001, default_reward)]#may returns to s0
                    else:  # rio
                        self.P[(s, a)] = [(s_next, 0.8, reward), (s + w, 0.2, default_reward)]

    @staticmethod
    def find_s_next(x, y, a, shape):
        h, w = shape
        s = x + (y * w)
        if a == 0:
            s_next = s - w
        elif a == 1:
            s_next = s + w
        elif a == 2:
            s_next = s + 1
        else:
            s_next = s - 1

        # cantos
        if x == 0 and a == 3:
            s_next = s
        elif x == w - 1 and a == 2:
            s_next = s
        elif y == 0 and a == 0:
            s_next = s
        elif y == h - 1 and a == 1:
            s_next = s

        return s_next

    def step(self, action):
        # Execute one time step within the environment
        random_number = random.uniform(0, 1)
        t_sum = 0.0
        # done = (self.current_s in self.G)
        for s_next, t, r in self.P[(self.current_s, action)]:
            t_sum += t
            if random_number <= t_sum:
                self.current_s = s_next
                done = (s_next in self.G)

                if self.state_as_img:
                    return self.state_img_cache[s_next], r, done, {}
                return s_next, r, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_s = self.s0
        if self.state_as_img:
            return self.state_img_cache[self.current_s]
        return self.current_s

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        if self.state_as_img:
            return self.state_img_cache[self.current_s]
        return self.current_s

    def find_s_from_img(self, state_img):
        h, w = self.shape
        for y in range(h):
            for x in range(w):
                s = x + (y * w)
                if (self.state_img_cache[s] == state_img).all():
                    return s

    @staticmethod
    def draw_policy(V, policy, shape, font_size, plot_value=False):

        values_reshaped = np.reshape(V, shape)
        if policy is not None:
            policy_reshaped = np.reshape(policy, shape)
        else:
            policy_reshaped = np.reshape(np.zeros(len(V)), shape)
        v_max = max(V)
        plt.figure(figsize=(10, 10))
        h, w = shape
        plt.imshow(values_reshaped, cmap='cool')
        ax = plt.gca()
        ax.set_xticks(np.arange(w) - .5)
        ax.set_yticks(np.arange(h) - .5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        a2uv = {0.0: (0, 1),  # north
                1.0: (0, -1),  # south
                2.0: (1, 0),  # east
                3.0: (-1, 0)  # west
                }

        for y in range(h):
            for x in range(w):

                if plot_value:
                    plt.text(x, y, round(values_reshaped[y, x], 2),
                             color='b', size=font_size / 2, verticalalignment='center',
                             horizontalalignment='center', fontweight='bold')
                else:
                    if y == h - 1 and x == w - 1:
                        plt.text(x, y, 'G',
                                 color='b', size=font_size, verticalalignment='center',
                                 horizontalalignment='center', fontweight='bold')
                    elif y == h - 1 and x != 0 and x != w - 1:  # cachoeira
                        plt.text(x, y, '-',
                                 color='b', size=font_size, verticalalignment='center',
                                 horizontalalignment='center', fontweight='bold')
                    else:
                        a = policy_reshaped[y, x]
                        if a is None: continue
                        if a < 0:
                            plt.text(x, y, ' ',
                                     color='b', size=font_size, verticalalignment='center',
                                     horizontalalignment='center', fontweight='bold')
                        else:
                            u, v = a2uv[a]
                            plt.arrow(x, y, u * .2, -v * .2, color='b', head_width=0.2, head_length=0.2)
        plt.grid(color='b', lw=2, ls='-')
        plt.show()

    @staticmethod
    def draw_img_state(shape, state):
        h, w = shape

        # adjusting image size to mat plot
        img_width = STATE_IMG_WIDTH/100

        fig = plt.figure(figsize=(img_width, img_width))
        ax = fig.add_subplot(111, aspect='equal')

        for y in range(h):
            for x in range(w):
                s = x + (y * w)

                # By default, the (0, 0) coordinate in matplotlib is the bottom left corner,
                # so we need to invert the y coordinate to plot the matrix correctly
                matplot_x = x
                matplot_y = h - y - 1

                if x == w - 1 and y == h - 1:  # meta
                    ax.add_patch(plt.Rectangle((matplot_x - 0.5, matplot_y - 0.5), 1, 1, facecolor='gray'))
                    # ax.add_patch(plt.Rectangle((matplot_x - 0.5, matplot_y - 0.5), 1, 1, fill=False, hatch='+'))
                elif x == 0:  # margem direita
                    text = 'D'
                elif x == w - 1:  # margem esquerda
                    text = 'E'
                elif y == 0:  # ponte
                    text = 'B'
                elif y == h - 1:  # cachoeira
                    ax.add_patch(plt.Rectangle((matplot_x - 0.5, matplot_y - 0.5), 1, 1, fill=False, hatch='xx'))
                else:  # rio
                    ax.add_patch(plt.Rectangle((matplot_x - 0.5, matplot_y - 0.5), 1, 1, fill=False, hatch='..'))

                if s == state:
                    ax.add_patch(plt.Circle((matplot_x, matplot_y), 0.4, facecolor='black'))

                # ax.annotate(str(s), xy=(matplot_x, matplot_y), ha='center', va='center')

        offset = .5
        ax.set_xlim(-offset, w - offset)
        ax.set_ylim(-offset, h - offset)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.hlines(y=np.arange(h + 1) - offset, xmin=-offset, xmax=w - offset, color='black')
        ax.vlines(x=np.arange(w + 1) - offset, ymin=-offset, ymax=h - offset, color='black')

        plt.savefig('environment/img/river_{}.png'.format(state))
        return cv2.imread('environment/img/river_{}.png'.format(state))

    @staticmethod
    def process_state_image(image):
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = image_gray.astype(float)
        image_gray /= 255.0
        return image_gray

    @staticmethod
    def print_result(policy, V, steps, updates, shape, font_size=30):
        if steps > 0:
            print('\n Passos', steps)
        if updates > 0:
            print('\n Atualizações', updates)

        print("\n Value")
        print(np.reshape(V, shape))

        if policy is not None:
            print("\n Policy")
            print(np.reshape(policy, shape))

        RiverCrossingEnv.draw_policy(V, policy, shape, font_size, True)
        if policy is not None:
            RiverCrossingEnv.draw_policy(V, policy, shape, font_size, False)

            def render(model):
                h, w = shape
                lineAction = ''
                lineState = ''
                for y in range(h):
                    for x in range(w):
                        state = x + (y * w)
                        q_s = model.find_qs(state)
                        action = np.argmax(q_s)
                        lineState = '{}\t{}'.format(lineState, str(round(max(q_s), 3)))
                        act = 'L'
                        if action == 0:
                            act = 'U'
                        elif action == 1:
                            act = 'D'
                        elif action == 2:
                            act = 'R'

                        if state in [19]:
                            act = 'G'
                        elif state in [17, 18]:
                            act = '-'
                        lineAction = '{}\t{}'.format(lineAction, act)
                    lineAction = '{}\n'.format(lineAction)
                    lineState = '{}\n'.format(lineState)
                print(lineAction)
                print('')
                print(lineState)


    def render(self, model):
        h, w = self.shape
        lineAction = ''
        lineState = ''

        for y in range(h):
            for x in range(w):
                state = x + (y * w)
                if self.state_as_img:
                    q_s = model.find_qs(self.state_img_cache[state])
                else:
                    q_s = model.find_qs(state)
                action = np.argmax(q_s)
                lineState = '{}\t{}'.format(lineState, str(round(max(q_s), 3)))
                act = 'L'
                if action == 0:
                    act = 'U'
                elif action == 1:
                    act = 'D'
                elif action == 2:
                    act = 'R'

                if state in self.G:
                    act = 'G'
                elif y == h - 1 and state != self.s0:
                    act = '-'

                lineAction = '{}\t{}'.format(lineAction, act)
            lineAction = '{}\n'.format(lineAction)
            lineState = '{}\n'.format(lineState)
        print(lineAction)
        print('')
        print(lineState)
