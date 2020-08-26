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
