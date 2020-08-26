import tensorflow as tf
import gym, time, random, threading
from gym import wrappers
from keras.models import *
from keras.layers import *
from keras.utils import plot_model
from keras import backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#--- constants
ENV = 'CartPole-v0'
env = gym.make(ENV)
NUM_STATES = env.observation_space.shape[0]
NUM_ACTIONS = env.action_space.n
NONE_STATE = np.zeros(NUM_STATES)

MIN_BATCH = 5
EPSILON = 0.2
LOSS_V = 0.2
LOSS_ENTROPY = 0.01
LEARNING_RATE = 2e-3

GAMMA = 0.99
N_STEP_RETURN = 5
GAMMA_N = GAMMA ** N_STEP_RETURN

N_WORKERS = 8
Tmax = 3 * N_WORKERS

EPS_START = 0.5
EPS_END = 0.0
EPS_STEPS = 200 * N_WORKERS

class Brain:
    def __init__(self):
        with tf.name_scope("brain"):
            self.train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
            K.set_session(SESS)
            self.model = self._build_model()
            self.opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
            self.prop_old = 1
            self.graph = self.build_graph()

    def _build_model(self):
        l_input = Input(batch_shape=(None, NUM_STATES))
        l_dense = Dense(16, activation='relu')(l_input)
        out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
        out_value = Dense(1, activation='linear')(l_dense)
        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        model._make_predict_function()
        plot_model(model, to_file='PPO.png', show_shapes=True)
        return model

    def build_graph(self):
        self.s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATES))
        self.a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        self.r_t = tf.placeholder(tf.float32, shape=(None, 1))
        p, v = self.model(self.s_t)

        # loss function
        advantage = tf.subtract(self.r_t, v)
        self.prob = tf.multiply(p, self.a_t) + 1e-10
        r_theta = tf.div(self.prob, self.prop_old)
        advantage_CPI = tf.multiply(r_theta, tf.stop_gradient(advantage))
        r_clip = tf.clip_by_value(r_theta, r_theta-EPSILON, r_theta+EPSILON)
        clipped_advantage_CPI = tf.multiply(r_clip, tf.stop_gradient(advantage))
        loss_CLIP = -tf.reduce_mean(tf.minimum(advantage_CPI, clipped_advantage_CPI), axis=1, keep_dims=True)
        loss_value = LOSS_V * tf.square(advantage)
        entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)
        self.loss_total = tf.reduce_mean(loss_CLIP + loss_value - entropy)
        minimize = self.opt.minimize(self.loss_total)
        return minimize

    def update_parameter_server(self):
        if len(self.train_queue[0]) < MIN_BATCH:
            return

        s, a, r, s_, s_mask = self.train_queue
        self.train_queue = [[], [], [], [], []]
        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        _, v = self.model.predict(s_)
        r = r + GAMMA_N * v * s_mask
        feed_dict = {self.s_t: s, self.a_t: a, self.r_t: r}

        minimize = self.graph
        SESS.run(minimize, feed_dict)
        self.prob_old = self.prob

    def predict_p(self, s):
        p, v = self.model.predict(s)
        return p

    def train_push(self, s, a, r, s_):
        self.train_queue[0].append(s)
        self.train_queue[1].append(a)
        self.train_queue[2].append(r)

        if s_ is None:
            self.train_queue[3].append(NONE_STATE)
            self.train_queue[4].append(0.)
        else:
            self.train_queue[3].append(s_)
            self.train_queue[4].append(1.)

class Agent:
    def __init__(self, brain):
        self.brain = brain
        self.memory = []
        self.R = 0.

    def act(self, s):
        if frames >= EPS_STEPS:
            eps = EPS_END
        else:
            eps = EPS_START + frames * (EPS_END - EPS_START) / EPS_STEPS

        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS - 1)
        else:
            s = np.array([s])
            p = self.brain.predict_p(s)
            a = np.random.choice(NUM_ACTIONS, p=p[0])
            return a

    def advantage_push_brain(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]
            return s, a, self.R, s_

        a_cats = np.zeros(NUM_ACTIONS)
        a_cats[a] = 1
        self.memory.append((s, a_cats, r, s_))
        self.R = (self.R + r * GAMMA_N) / GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                self.brain.train_push(s, a, r, s_)
                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)
            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            self.brain.train_push(s, a, r, s_)
            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)

class Environment:
    total_reward_vec = np.zeros(10)
    count_trial_each_thread = 0

    def __init__(self, name, thread_type, brain):
        self.name = name
        self.thread_type = thread_type
        self.env = gym.make(ENV)
        self.agent = Agent(brain)

    def run(self):
        global frames
        global isLearned

        if (self.thread_type is 'test') and (self.count_trial_each_thread == 0):
            self.env.reset()

        s = self.env.reset()
        R = 0
        step = 0
        while True:
            if self.thread_type is 'test':
                self.env.render()
                time.sleep(0.1)

            a = self.agent.act(s)
            s_, r, done, info = self.env.step(a)
            step += 1
            frames += 1

            r = 0
            if done:
                s_ = None
                if step < 199:
                    r = -1
                else:
                    r = 1

            self.agent.advantage_push_brain(s, a, r, s_)

            s = s_
            R += r
            if done or (frames % Tmax == 0):
                if not(isLearned) and self.thread_type is 'learning':
                    self.agent.brain.update_parameter_server()

            if done:
                self.total_reward_vec = np.hstack((self.total_reward_vec[1:], step))
                self.count_trial_each_thread += 1
                break

        print("Thread: " + self.name + ", Trial: " + str(self.count_trial_each_thread) + ", Step: " + str(step) + ", Step(mean): " + str(self.total_reward_vec.mean()))

        if self.total_reward_vec.mean() > 199:
            isLearned = True
            time.sleep(2.0)

class WorkerThread:
    def __init__(self, thread_name, thread_type, brain):
        self.environment = Environment(thread_name, thread_type, brain)
        self.thread_type = thread_type

    def run(self):
        while True:
            if not(isLearned) and self.thread_type is 'learning':
                self.environment.run()
            if not(isLearned) and self.thread_type is 'test':
                time.sleep(1.0)
            if isLearned and self.thread_type is 'learning':
                time.sleep(3.0)
            if isLearned and self.thread_type is 'test':
                time.sleep(3.0)
                self.environment.run()
