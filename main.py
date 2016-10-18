import random
import os
#
# os.environ['GLOG_minloglevel'] = '2'

import cv2
import numpy as np
import game.wrapped_flappy_bird as game_interface
import caffe
from collections import deque

ACTIONS = 2
MAX_TRANSITION = 50000
OBSERVE_TRANSITION = 100000
BATCH_SIZE = 32
GAMMA = 0.99 
EPSILON_ANNEALING_TIME = 1000000
EPSILON_BEGIN = 1
EPSILON_END = 0.0001
EPSILON_SLOPE = -(EPSILON_BEGIN - EPSILON_END) / EPSILON_ANNEALING_TIME
ACTION_PROBABILITY = 0.1
STATE_FRAME = 4
UPDATE_STEP = 1000
UPDATE_GAMMA = 1

class Dqn:
    def __init__(self, model=None):
        solver_file = 'dqn_solver.prototxt'
        self.solver = caffe.AdamSolver(solver_file)
        # self.solver = caffe.SGDSolver(solver_file)
        self.net = self.solver.net
        self.target = caffe.Net('dqn.prototxt', caffe.TEST)

        if model:
            self.net.copy_from(model)

        self.epsilon = EPSILON_BEGIN
        self.steps = 0
        self.experience_replay = deque()
        # self.old_conv = 0

    def train(self):
        mini_batch = random.sample(self.experience_replay, BATCH_SIZE)

        state_batch = np.array([data[0] for data in mini_batch])
        action_batch = np.array([data[1] for data in mini_batch])
        reward_batch = np.array([data[2] for data in mini_batch])
        new_state_batch = np.array([data[3] for data in mini_batch])
        terminal_batch = np.array([data[4] for data in mini_batch])

        self.target.blobs['frames'].data[...] = new_state_batch
        self.target.blobs['action'].data[...] = action_batch
        self.target.forward(end='reduction')
        new_q_batch = self.target.blobs['reduction'].data.copy()

        target_batch = []
        for idx, data in enumerate(mini_batch):
            target = data[2] + GAMMA * new_q_batch[idx] * (1 - data[4])
            target_batch.append(target)  

        self.net.blobs['frames'].data[...] = state_batch
        self.net.blobs['action'].data[...] = action_batch
        self.net.blobs['target'].data[...] = np.reshape(target_batch, [BATCH_SIZE, 1])
        self.solver.step(1)
        print "steps: ", self.steps, "\tloss: ", self.net.blobs['loss'].data[...], "\tepsilon: ", self.epsilon

    def get_action(self):
        action = np.zeros(ACTIONS)
        if random.random() < self.epsilon:
            action_index = 1 if (random.random() < ACTION_PROBABILITY) else 0
            action[action_index] = 1
        else:
            state = np.zeros([BATCH_SIZE, STATE_FRAME, 80, 80])
            state = np.append(state[:BATCH_SIZE - 1, :, :, :], np.reshape(self.state, [1, STATE_FRAME, 80, 80]), axis=0)

            self.net.blobs['frames'].data[...] = state
            output = self.net.forward(end='fc2')
            # new_conv = self.net.blobs['pool1'].data[...]
            # print np.array_equal(new_conv, self.old_conv)
            print"Q-value: ", output['fc2'][-1], "\taction: ", np.argmax(output['fc2'][-1])
            action[np.argmax(output['fc2'][-1])] = 1

        if self.epsilon > EPSILON_END and self.steps > OBSERVE_TRANSITION:
            self.epsilon += EPSILON_SLOPE

        # self.old_conv = new_conv.copy()
        return action

    def save_transition(self, frame, action, reward, terminal):
        new_state = np.append(self.state[:STATE_FRAME - 1,:,:], frame, axis=0)
        self.experience_replay.append((self.state, action, reward, new_state, terminal))
        if len(self.experience_replay) > MAX_TRANSITION:
            self.experience_replay.popleft()
        
        if self.steps > OBSERVE_TRANSITION:
            self.train()
            if self.steps % UPDATE_STEP == 0:
                self.update_target()

        if self.steps % 1000 == 0:
            print "STEP: %d" % self.steps

        self.state = new_state
        self.steps += 1

    def set_initial_state(self, frame):
        self.state = np.stack((frame for _ in range(STATE_FRAME)), axis = 0)

    def update_target(self):
        for layer in ['conv1', 'conv2', 'conv3', 'fc1', 'fc2']:
            self.target.params[layer][0].data[...]\
                = self.target.params[layer][0].data * (1 - UPDATE_GAMMA) + \
                  self.net.params[layer][0].data * UPDATE_GAMMA
            self.target.params[layer][0].data[...] \
                = self.target.params[layer][0].data * (1 - UPDATE_GAMMA) + \
                  self.net.params[layer][0].data * UPDATE_GAMMA


def preprocess(image):
    gray_image = cv2.cvtColor(cv2.resize(image, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(binary_image,(1, 80, 80))


def playgame():
    dqn = Dqn()
    flappy_bird = game_interface.GameState()
    initial_action = np.array([1, 0])
    initial_frame, reward, terminal = flappy_bird.frame_step(initial_action)

    # initial
    initial_frame = cv2.cvtColor(cv2.resize(initial_frame, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, initial_frame = cv2.threshold(initial_frame,1,255,cv2.THRESH_BINARY)

    dqn.set_initial_state(initial_frame)
    while True:
        action = dqn.get_action()
        frame, reward, terminal = flappy_bird .frame_step(action)

        sample = preprocess(frame)
        dqn.save_transition(sample, action, reward, terminal)
        # if terminal:
        #     dqn.set_initial_state(initial_frame)

playgame()

