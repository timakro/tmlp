#!/usr/bin/env python
import os
import signal
import subprocess
import time
import shutil
import logging
import re
import math
from gzip import GzipFile

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed, Embedding, ZeroPadding2D, Conv2D, Reshape, LSTM, Dense


SERVER_AGENTS = [2]*2
EPISODE_LENGTH = 200 #7500
SEQUENCE_LENGTH = 100 #300 # Internet says 200-300, 200-400

# Hyper parameter guidance: https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
GAMMA = 0.99
LAMBD = 0.95
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 5.0 # Internet says 1-10, 40, 0.5
LEARNING_RATE = 1e-4

LSTM_UNITS = 512
TILE_COUNT = 22

ACTION_DTYPE = np.dtype([
    ('target_x', np.float32),
    ('target_y', np.float32),
    ('b_left', np.bool),
    ('b_right', np.bool),
    ('b_jump', np.bool),
    ('b_fire', np.bool),
    ('b_hook', np.bool),
    ('weapon', np.uint8),
    ('value', np.float32),
    ('reward', np.float32),
])


class A2CNetwork:

    def __init__(self, sess):
        self.sess = sess

        self.input = tf.placeholder(tf.uint8, shape=[None, None, 50, 90], name='input')
        self.rnn_state_in = tf.placeholder(tf.float32, shape=[None, LSTM_UNITS*2], name='rnn_state_in')
        batch_size = tf.shape(self.input)[0]
        seq_length = tf.shape(self.input)[1]
        net = Embedding(TILE_COUNT, 8)(self.input)
        net = TimeDistributed(ZeroPadding2D(padding=(2, 0)))(net)
        net = TimeDistributed(Conv2D(filters=32, kernel_size=9, strides=3, padding='same', activation='relu'))(net)
        net = TimeDistributed(Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation='relu'))(net)
        net = TimeDistributed(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))(net)
        net = tf.reshape(net, [batch_size, seq_length, np.product(net.get_shape()[2:])])
        net, *rnn_state_out = LSTM(LSTM_UNITS, return_sequences=True, return_state=True)(net, initial_state=tf.split(self.rnn_state_in, 2, axis=1))
        self.rnn_state_out = tf.concat(rnn_state_out, axis=1, name='rnn_state_out')
        #net = Dense(4096, activation='relu')(net)
        self.target_mu = Dense(2, activation='tanh', name='target_mu')(net)
        self.target_var = Dense(2, activation='softplus', name='target_var')(net)
        self.binary = Dense(5, activation='sigmoid', name='binary')(net)
        self.weapon = Dense(5, activation='softmax', name='weapon')(net)
        self.value = Dense(1, name='value')(net)

        # Loss and train ops
        self.target_acts = tf.placeholder(tf.float32, shape=[None, None, 2])
        self.binary_acts = tf.placeholder(tf.bool, shape=[None, None, 5])
        self.weapon_acts = tf.placeholder(tf.int32, shape=[None, None])
        self.returns = tf.placeholder(tf.float32, shape=[None, None])
        self.advantages = tf.placeholder(tf.float32, shape=[None, None])

        target_dist = tf.distributions.Normal(loc=self.target_mu, scale=tf.sqrt(self.target_var))
        target_logprob = tf.reduce_sum(target_dist.log_prob(self.target_acts), axis=2)
        target_entropy = tf.reduce_sum(target_dist.entropy(), axis=2)

        binary_dist = tf.distributions.Bernoulli(probs=self.binary, dtype=tf.bool)
        binary_logprob = tf.reduce_sum(binary_dist.log_prob(self.binary_acts), axis=2)
        binary_entropy = tf.reduce_sum(binary_dist.entropy(), axis=2)

        weapon_dist = tf.distributions.Categorical(probs=self.weapon)
        weapon_logprob = weapon_dist.log_prob(self.weapon_acts)
        weapon_entropy = weapon_dist.entropy()

        logprob = target_logprob + binary_logprob + weapon_logprob
        entropy = target_entropy + binary_entropy + weapon_entropy

        policy_loss = tf.reduce_mean(-logprob * self.advantages)
        value_loss = tf.losses.mean_squared_error(self.returns, tf.squeeze(self.value, axis=2))
        entropy = tf.reduce_mean(entropy)

        loss = policy_loss + value_loss*VALUE_COEF - entropy*ENTROPY_COEF

        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, MAX_GRAD_NORM)
        self.train_op = optimizer.apply_gradients(zip(gradients, variables))

        # Init and save/restore ops
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
        try: shutil.rmtree('savedmodel')
        except FileNotFoundError: pass
        tf.saved_model.simple_save(
            self.sess, 'savedmodel', inputs={'input': self.input, 'rnn_state_in': self.rnn_state_in},
            outputs={'target_mu': self.target_mu, 'target_var': self.target_var, 'binary': self.binary,
                     'weapon': self.weapon, 'rnn_state_out': self.rnn_state_out, 'value': self.value})

    def train(self, inputs, rnn_state_in, target_acts, binary_acts, weapon_acts, returns, advantages):
        _, rnn_state_out = self.sess.run([self.train_op, self.rnn_state_out],
                feed_dict={self.input: inputs, self.rnn_state_in: rnn_state_in,
                           self.target_acts: target_acts, self.binary_acts: binary_acts, self.weapon_acts: weapon_acts,
                           self.returns: returns, self.advantages: advantages})
        return rnn_state_out

    def forward_pass(self):
        batch = np.random.random([1, 1, 50, 90])
        rnn_state_in = np.random.random([1, LSTM_UNITS*2])

        return self.sess.run([self.target_mu, self.target_var, self.binary, self.weapon, self.state_out],
                feed_dict={self.input: batch, self.rnn_state_in: rnn_state_in})

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

    while running:
        print("Episode", episode)

        # Start game servers
        runners = []
        for i, num_agents in enumerate(SERVER_AGENTS):
            cmds = ['sv_port {};'.format(8303 + i)
                   ,'sv_name TMLP Training Srv {};'.format(i)
                   ,'dbg_dummies {};'.format(num_agents)
                   ,'tmlp_server_id {};'.format(i)
                   ,'tmlp_lstm_units {};'.format(LSTM_UNITS)
                   ,'tmlp_sequence_length {};'.format(SEQUENCE_LENGTH)]
            p = subprocess.Popen(['teeworlds-tmlp/build/teeworlds_srv', ''.join(cmds)],
                                 stdout=subprocess.PIPE, universal_newlines=True,
                                 preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN))
            runners.append(p)

        rnn_state = np.zeros([sum(SERVER_AGENTS), LSTM_UNITS*2], dtype=np.float32)
        for _ in range(EPISODE_LENGTH // SEQUENCE_LENGTH):
            # Game servers simulate one sequence
            net.write_model_to_disk()
            for p in runners:
                p.send_signal(signal.SIGUSR1)
            for p in runners:
                while p.stdout.readline() != "Rollout saved to disk.\n":
                    pass

            # Read rollouts from disk
            states = np.empty([sum(SERVER_AGENTS), SEQUENCE_LENGTH, 50, 90], dtype=np.uint8)
            target_acts = np.empty([sum(SERVER_AGENTS), SEQUENCE_LENGTH, 2], dtype=np.float32)
            binary_acts = np.empty([sum(SERVER_AGENTS), SEQUENCE_LENGTH, 5], dtype=np.bool)
            weapon_acts = np.empty([sum(SERVER_AGENTS), SEQUENCE_LENGTH], dtype=np.uint8)
            values = np.empty([sum(SERVER_AGENTS), SEQUENCE_LENGTH], dtype=np.float32)
            rewards = np.empty([sum(SERVER_AGENTS), SEQUENCE_LENGTH], dtype=np.float32)
            for i, (s, a) in enumerate((s, a) for s, num_agents in enumerate(SERVER_AGENTS) for a in range(num_agents)):
                path = 'rollouts/{:04}-{:02}'.format(s, a)
                with GzipFile(path+'-state.gz') as state_file, GzipFile(path+'-action.gz') as action_file:
                    states[i] = np.frombuffer(state_file.read(), dtype=np.uint8).reshape([SEQUENCE_LENGTH, 50, 90])
                    action_data = np.frombuffer(action_file.read(), dtype=ACTION_DTYPE)
                    target_acts[i] = structured_to_unstructured(action_data[['target_x', 'target_y']])
                    binary_acts[i] = structured_to_unstructured(action_data[['b_left', 'b_right', 'b_jump', 'b_fire', 'b_hook']])
                    weapon_acts[i] = action_data['weapon']
                    values[i] = action_data['value']
                    rewards[i] = action_data['reward']
                os.unlink(path+'-state.gz')
                os.unlink(path+'-action.gz')

            # Calculate return and advantage
            returns = np.empty([sum(SERVER_AGENTS), SEQUENCE_LENGTH], dtype=np.float32)
            #advantages = np.empty([sum(SERVER_AGENTS), SEQUENCE_LENGTH], dtype=np.float32)
            future_reward = values[:,-1]
            for i in reversed(range(SEQUENCE_LENGTH)):
                returns[:,i] = future_reward = rewards[:,i] + GAMMA*future_reward
            advantages = returns - values

            rnn_state = net.train(states, rnn_state, target_acts, binary_acts, weapon_acts, returns, advantages)

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

    # Workaround for https://github.com/tensorflow/tensorflow/issues/23780
    from tensorflow.core.protobuf import rewriter_config_pb2
    config_proto = tf.ConfigProto()
    off = rewriter_config_pb2.RewriterConfig.OFF
    config_proto.graph_options.rewrite_options.arithmetic_optimization = off

    with tf.Session(config=config_proto) as sess:
        main(sess)
