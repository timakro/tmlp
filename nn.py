#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed, Embedding, ZeroPadding2D, Conv2D, Reshape, LSTM, Dense


BATCH_SIZE = 42
SEQUENCE_LENGTH = 300 # Sources say 200-400 or 200-300 is feasible. 300 frames is 12 seconds.

TILE_COUNT = 22
WEAPON_COUNT = 5

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

    def __init__(self, sess, live=False):
        self.sess = sess
        self.batch_size = 1 if live else BATCH_SIZE
        self.seq_length = 1 if live else SEQUENCE_LENGTH
        self.lstm_units = 512

        self.input = tf.placeholder(tf.uint8, shape=[self.batch_size, self.seq_length, 50, 90], name='input')
        self.state_in = tf.zeros([self.batch_size, self.lstm_units*2], name='state_in')
        net = Embedding(TILE_COUNT, 8)(self.input)
        net = TimeDistributed(ZeroPadding2D(padding=(2, 0)))(net)
        net = TimeDistributed(Conv2D(filters=32, kernel_size=9, strides=3, padding='same', activation='relu'))(net)
        net = TimeDistributed(Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation='relu'))(net)
        net = TimeDistributed(Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu'))(net)
        net = tf.reshape(net, [self.batch_size, self.seq_length, -1])
        net, *state_out = LSTM(self.lstm_units, return_sequences=True, return_state=True)(net, initial_state=tf.split(self.state_in, 2, axis=1))
        self.state_out = tf.concat(state_out, axis=1, name='state_out')
        #net = Dense(4096, activation='relu')(net)
        self.target_mu = Dense(2, activation='tanh', name='target_mu')(net)
        self.target_var = Dense(2, activation='softplus', name='target_var')(net)
        self.binary = Dense(5, activation='sigmoid', name='binary')(net)
        self.weapon = Dense(WEAPON_COUNT, activation='softmax', name='weapon')(net)

        sess.run(tf.global_variables_initializer())

    def save_model(self):
        tf.saved_model.simple_save(
            self.sess, 'testsave', inputs={'input': self.input, 'state_in': self.state_in},
            outputs={'target_mu': self.target_mu, 'target_var': self.target_var, 'binary': self.binary, 'weapon': self.weapon, 'state_out': self.state_out})

    def forward_pass(self):
        batch = np.random.random([self.batch_size, self.seq_length, 50, 90])
        state_in = np.random.random([self.batch_size, 512*2])

        self.sess.run([self.target_mu, self.target_var, self.binary, self.weapon, self.state_out],
                      feed_dict={self.input: batch})

        import time
        before = time.time()
        for i in range(100):
            self.sess.run([self.target_mu, self.target_var, self.binary, self.weapon, self.state_out],
                          feed_dict={self.input: batch})
        print((time.time()-before)/100)




if __name__ == '__main__':
    with tf.Session() as sess:
        net = A2CNetwork(sess, live=True)
        print(net.forward_pass())
        #net.save_model()
