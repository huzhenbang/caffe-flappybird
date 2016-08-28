import random

import cv2
import numpy as np
import game.wrapped_flappy_bird as game_interface
import caffe
from collections import deque 

ACTIONS = 2
MAX_TRANSITION = 1000
OBSERVE_TRANSITION = 500
BATCH_SIZE = 32
GAMMA = 0.99 
EPSILON_ANNEALING_TIME = 50000
EPSILON_BEGIN = 1
EPSILON_END = 0.0001
EPSILON_SLOPE = -(EPSILON_BEGIN - EPSILON_END) / EPSILON_ANNEALING_TIME
ACTION_PROBABILITY = 0.1
STATE_FRAME = 4

class Dqn:
    def __init__(self, model=None):
        solver_file = 'dqn_solver.prototxt'
        self.solver = caffe.SGDSolver(solver_file)
        self.net = self.solver.net

        if model:
            self.net.copy_from(model)

        self.epsilon = EPSILON_BEGIN
        self.steps = 0
        self.experience_replay = deque()

    def train(self):
        mini_batch = random.sample(self.experience_replay, BATCH_SIZE)

        state_batch = np.array([data[0] for data in mini_batch])
        action_batch = np.array([data[1] for data in mini_batch])
        reward_batch = np.array([data[2] for data in mini_batch])
        new_state_batch = np.array([data[3] for data in mini_batch])
        terminal_batch = np.array([data[4] for data in mini_batch])

        self.net.blobs['frames'].data[...] = new_state_batch
        self.net.forward(end='fc2')
        new_q_batch = self.net.blobs['fc2'].data.copy()
        self.net.blobs['frames'].data[...] = state_batch
        self.net.forward(end='fc2')
        q_batch = self.net.blobs['fc2'].data.copy()

        label_batch = []
        for idx, data in enumerate(mini_batch):
            target = data[2] + GAMMA * np.max(new_q_batch[idx]) * (1 - data[4])
            # print target
            # label = np.zeros([ACTIONS])
            label = q_batch[idx]
            label[np.argmax(data[1])] = target

            label_batch.append(label)   

        self.net.blobs['frames'].data[...] = state_batch
        self.net.blobs['action'].data[...] = label_batch
        self.solver.step(1)

    def get_action(self):
        action =  np.zeros(ACTIONS)
        if random.random() < self.epsilon:
            action_index = 1 if (random.random() < ACTION_PROBABILITY) else 0
            action[action_index] = 1
        else:
            state = np.zeros([BATCH_SIZE, STATE_FRAME, 80, 80])
            state = np.append(state[:BATCH_SIZE - 1, :, :, :], np.reshape(self.state, [1, STATE_FRAME, 80, 80]), axis=0)
            self.net.blobs['frames'].data[...] = state
            output = self.net.forward(end='fc2')
            print "network output action:", output['fc2'][-1], np.argmax(output['fc2'][-1])
            action[np.argmax(output['fc2'][-1])] = 1

        if self.epsilon > EPSILON_END and self.steps > OBSERVE_TRANSITION:
            self.epsilon += EPSILON_SLOPE

        return action

    def save_transition(self, frame, action, reward, terminal):
        new_state = np.append(self.state[:STATE_FRAME - 1,:,:], frame, axis=0)
        self.experience_replay.append((self.state, action, reward, new_state, terminal))
        if len(self.experience_replay) > MAX_TRANSITION:
            self.experience_replay.popleft()
        
        if self.steps > OBSERVE_TRANSITION:
            self.train()
        self.state = new_state
        self.steps += 1

    def set_initial_state(self, frame):
        self.state = np.stack((frame for _ in range(STATE_FRAME)), axis = 0)

def preprocess(image):
    gray_image = cv2.cvtColor(cv2.resize(image, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(binary_image,(1, 80, 80))

def playgame():
    dqn = Dqn("caffe_dqn_train_iter_3000.caffemodel")
    # dqn = Dqn()
    flappy_bird = game_interface.GameState()
    initial_action = np.array([1, 0])
    initial_frame, reward, terminal = flappy_bird.frame_step(initial_action)

    #initial
    initial_frame = cv2.cvtColor(cv2.resize(initial_frame, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, initial_frame = cv2.threshold(initial_frame,1,255,cv2.THRESH_BINARY)

    dqn.set_initial_state(initial_frame)
    while True:
        action = dqn.get_action()
        frame, reward, terminal = flappy_bird .frame_step(action)

        sample = preprocess(frame)
        dqn.save_transition(sample, action, reward, terminal)
        if terminal:
            dqn.set_initial_state(initial_frame)

playgame()

