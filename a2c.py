#!/usr/bin/env python
import os
import signal
import subprocess
import time
import shutil
import logging
import re

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed, Embedding, ZeroPadding2D, Conv2D, Reshape, LSTM, Dense


NUM_GAME_SERVERS = 2
SEQUENCE_LENGTH = 300 # Sources say 200-400 or 200-300 is feasible. 300 frames is 12 seconds.

LSTM_UNITS = 512

TILE_COUNT = 22

LABEL_DTYPE = np.dtype([
    ('targetx', np.int32),
    ('targety', np.int32),
    ('direction', np.int8),
    ('weapon', np.int8),
    ('jump', np.bool),
    ('fire', np.bool),
    ('hook', np.bool),
])


class A2CNetwork:

    def __init__(self, sess):
        self.sess = sess

        self.input = tf.placeholder(tf.uint8, shape=[None, None, 50, 90], name='input')
        self.state_in = tf.placeholder(tf.float32, shape=[None, LSTM_UNITS*2], name='state_in')
        seq_length = tf.shape(self.input)[1]
        net = Embedding(TILE_COUNT, 8)(self.input)
        net = TimeDistributed(ZeroPadding2D(padding=(2, 0)))(net)
        net = TimeDistributed(Conv2D(filters=32, kernel_size=9, strides=3, padding='same', activation='relu'))(net)
        net = TimeDistributed(Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation='relu'))(net)
        net = TimeDistributed(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))(net)
        net = tf.reshape(net, [-1, seq_length, np.product(net.get_shape()[2:])])
        net, *state_out = LSTM(LSTM_UNITS, return_sequences=True, return_state=True)(net, initial_state=tf.split(self.state_in, 2, axis=1))
        self.state_out = tf.concat(state_out, axis=1, name='state_out')
        #net = Dense(4096, activation='relu')(net)
        self.target_mu = Dense(2, activation='tanh', name='target_mu')(net)
        self.target_var = Dense(2, activation='softplus', name='target_var')(net)
        self.binary = Dense(5, activation='sigmoid', name='binary')(net)
        self.weapon = Dense(5, activation='softmax', name='weapon')(net)
        self.value = Dense(1, name='value')(net)

        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1, pad_step_number=True)

    def load_variables(self):
        ckpt_state = tf.train.get_checkpoint_state('checkpoints')
        if ckpt_state:
            self.saver.recover_last_checkpoints(ckpt_state.all_model_checkpoint_paths)
            self.saver.restore(self.sess, ckpt_state.model_checkpoint_path)
            episode = int(re.search(r'\d+', ckpt_state.model_checkpoint_path).group()) + 1
        else:
            self.sess.run(self.init_op)
            episode = 1
        return episode

    def save_variables(self, episode):
        self.saver.save(self.sess, 'checkpoints/episode', global_step=episode, write_meta_graph=False)

    def write_model_to_disk(self):
        shutil.rmtree('savedmodel')
        tf.saved_model.simple_save(
            self.sess, 'savedmodel', inputs={'input': self.input, 'state_in': self.state_in},
            outputs={'target_mu': self.target_mu, 'target_var': self.target_var, 'binary': self.binary,
                     'weapon': self.weapon, 'state_out': self.state_out, 'value': self.value})

    def forward_pass(self):
        batch = np.random.random([1, 1, 50, 90])
        state_in = np.random.random([1, 512*2])

        return self.sess.run([self.target_mu, self.target_var, self.binary, self.weapon, self.state_out],
                feed_dict={self.input: batch, self.state_in: state_in})

    def time_forward_pass(self):
        self.forward_pass() # First pass takes longer
        before = time.time()
        for i in range(500):
            self.forward_pass()
        timespan = (time.time() - before) / 500
        print("Forward pass takes {} seconds".format(timespan))


def main(sess):
    # Handle exit signals
    running = True
    def schedule_exit(signum, frame):
        nonlocal running
        print("Finishing episode ... please wait for exit")
        running = False
    signal.signal(signal.SIGTERM, schedule_exit)
    signal.signal(signal.SIGINT, schedule_exit)

    # Load latest checkpoint
    net = A2CNetwork(sess)
    episode = net.load_variables()

    # Start game servers
    runners = []
    for i in range(NUM_GAME_SERVERS):
        cmds = ['sv_port {};'.format(8303 + i)
               ,'sv_name TMLP Training Srv {};'.format(i)
               ,'tmlp_server_id {};'.format(i)
               ,'tmlp_lstm_units {};'.format(LSTM_UNITS)
               ,'tmlp_episode_steps {};'.format(SEQUENCE_LENGTH)]
        p = subprocess.Popen(['teeworlds-tmlp/build/teeworlds_srv', ''.join(cmds)],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True,
                             preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
        runners.append(p)

    while running:
        # Game servers simulate one episode
        net.write_model_to_disk()
        for p in runners:
            p.send_signal(signal.SIGUSR1)
        for p in runners:
            while p.stdout.readline() != "Rollout saved to disk.\n":
                pass

        # TODO: Load rollouts
        # TODO: Train

        # Save checkpoint
        net.save_variables(episode)
        episode += 1

    for p in runners:
        p.terminate()


if __name__ == '__main__':
    # Change directory to script location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Hide tensorflow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

    with tf.Session() as sess:
        main(sess)

        #net = A2CNetwork(sess)
        #print(net.forward_pass())
        #net.time_forward_pass()
        #net.save_to_disk()
