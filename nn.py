#!/usr/bin/env python3
import os
import gzip
import random
import math
import json

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import tensorflow as tf
from tensorflow.keras.layers import (Input, TimeDistributed, Embedding, ZeroPadding2D, Conv2D,
                                     MaxPooling2D, Reshape, LSTM, Dropout, Dense)


EPOCHS = 999
BATCH_SIZE = 99 # I've got 99 samples but a batch ain't exceeding int32 max
SEQUENCE_LENGTH = 300 # Sources say 200-400 or 200-300 is feasible. 300 frames is 12 seconds.
VALIDATION_SPLIT = 0.2

SESSION_DIR = 'sessions'
CHECKPOINT_DIR = 'checkpoints'

TILE_COUNT = 22
WEAPON_COUNT = 6

LABEL_DTYPE = np.dtype([
    ('targetx', np.int32),
    ('targety', np.int32),
    ('direction', np.int8),
    ('weapon', np.int8),
    ('jump', np.bool),
    ('fire', np.bool),
    ('hook', np.bool),
])


def get_session_dicts():
    session_dicts = []
    for filename in os.listdir(SESSION_DIR):
        filename = os.path.join(SESSION_DIR, filename, 'meta.json')
        with open(filename) as metafile:
            meta = json.load(metafile)
            meta['volume'] -= 1 # Actions only take effect in the next frame, offset by 1 time step
            session_dicts.append(meta)
    return session_dicts

def split_off_test_set(train_set):
    total_frames = sum(s['volume'] for s in train_set)
    validation_frames = int(total_frames * VALIDATION_SPLIT)

    # Shuffle for equal density or sort by volume for compactness
    random.shuffle(train_set)
    #train_set.sort(key=lambda s: s['volume'], reverse=True)

    test_set = []
    cut_off = 0
    while cut_off < validation_frames:
        cut_off += train_set[-1]['volume']
        test_set.append(train_set.pop())
    return test_set

def split_into_super_batches(session_dicts):
    session_dicts.sort(key=lambda s: s['volume'], reverse=True)
    dataset = []
    total_batches = 0
    for i in range(0, len(session_dicts), BATCH_SIZE):
        super_batch = session_dicts[i:i+BATCH_SIZE]
        max_frames = max(s['volume'] for s in super_batch)
        batches = math.ceil(max_frames / SEQUENCE_LENGTH)
        dataset.append((super_batch, batches))
        total_batches += batches
    return dataset, total_batches

def iterate_batches(dataset):
    for super_batch, batches in dataset:
        x_files = [gzip.GzipFile(os.path.join(SESSION_DIR, s['sessionid'], 'data.gz')) for s in super_batch]
        y_files = [gzip.GzipFile(os.path.join(SESSION_DIR, s['sessionid'], 'labels.gz')) for s in super_batch]
        # Actions only take effect in the next frame, offset by 1 time step
        for y_file in y_files:
            y_file.read(LABEL_DTYPE.itemsize)
        for super_batch_index in range(batches):
            x = np.zeros((BATCH_SIZE, SEQUENCE_LENGTH, 50, 90), dtype=np.uint8)
            y_target = np.zeros((BATCH_SIZE, SEQUENCE_LENGTH, 2), dtype=np.float32)
            y_binary = np.zeros((BATCH_SIZE, SEQUENCE_LENGTH, 5), dtype=np.int8)
            y_weapon = np.zeros((BATCH_SIZE, SEQUENCE_LENGTH, 1), dtype=np.int8)
            mask = np.zeros((BATCH_SIZE, SEQUENCE_LENGTH), dtype=np.float32)
            for i, x_file in enumerate(x_files):
                data = x_file.read(90*50*SEQUENCE_LENGTH)
                data = np.reshape(np.frombuffer(data, dtype=np.uint8), (-1, 50, 90))
                x[i,:len(data)] = data
            for i, y_file in enumerate(y_files):
                data = y_file.read(LABEL_DTYPE.itemsize*SEQUENCE_LENGTH)
                data = np.frombuffer(data, dtype=LABEL_DTYPE)
                # Can't aim to center, normalize vector
                target = structured_to_unstructured(data[['targetx', 'targety']])
                invalid_target_mask = np.all(target == [0, 0], axis=1)
                target[invalid_target_mask] = [0, -1]
                y_target[i,:len(data)] = target / np.linalg.norm(target, axis=1, keepdims=True)
                # Encode direction as left/right buttons
                y_binary[i,:len(data)][data['direction']==-1,0] = 1
                y_binary[i,:len(data)][data['direction']==1,1] = 1
                y_binary[i,:len(data),2:] = structured_to_unstructured(data[['jump', 'fire', 'hook']])
                # Clip weapon to possible values
                y_weapon[i,:len(data),0] = np.clip(data['weapon'], 0, WEAPON_COUNT-1)
                # Set mask based on y instead of x because it's shorter by 1 time step
                mask[i,:len(data)] = 1.
            y = [y_target, y_binary, y_weapon]
            mask = [mask, mask, mask]
            yield super_batch_index, x, y, mask

def state_control(model, batches):
    for batch, (super_batch_index, x, y, mask) in enumerate(batches):
        # New super batch starts, reset states
        if super_batch_index == 0:
            model.reset_states()
        yield batch, x, y, mask
    # Reset states when we're done
    model.reset_states()

def build_model(live=False):
    inp = Input(batch_shape=(1 if live else BATCH_SIZE, 1 if live else SEQUENCE_LENGTH, 50, 90))
    net = Embedding(TILE_COUNT, 8)(inp)
    #net = TimeDistributed(ZeroPadding2D(padding=1))(net)
    net = TimeDistributed(Conv2D(filters=16, kernel_size=5, padding='same', activation='relu'))(net)
    net = TimeDistributed(MaxPooling2D(pool_size=2, strides=2))(net)
    net = TimeDistributed(Conv2D(filters=64, kernel_size=5, padding='same', activation='relu'))(net)
    #net = TimeDistributed(MaxPooling2D(pool_size=2, strides=2))(net)
    net = Reshape((1 if live else SEQUENCE_LENGTH, -1))(net)
    net = LSTM(32, return_sequences=True, stateful=True)(net)
    net = Dropout(0.2)(net)
    net = LSTM(128, return_sequences=True, stateful=True)(net)
    net = Dropout(0.2)(net)
    net = TimeDistributed(Dense(2048, activation='relu'))(net)
    net = Dropout(0.2)(net)
    net = TimeDistributed(Dense(2048, activation='relu'))(net)
    fin = Dropout(0.2)(net)
    target = TimeDistributed(Dense(2), name='target')(fin)
    binary = TimeDistributed(Dense(5, activation='sigmoid'), name='binary')(fin)
    weapon = TimeDistributed(Dense(WEAPON_COUNT, activation='softmax'), name='weapon')(fin)

    model = tf.keras.Model(inputs=inp, outputs=[target, binary, weapon])

    # outputs: target, binary, weapon
    model.compile(optimizer='adam',
                  loss=[tf.keras.losses.mean_squared_error, 'binary_crossentropy', 'sparse_categorical_crossentropy'], # Keras MSE because of bug with sample weights
                  loss_weights=[2.0, 1.5, 0.5], # Balance losses for a random model
                  sample_weight_mode='temporal',
                  weighted_metrics={'target': 'mean_absolute_error', 'binary': 'accuracy', 'weapon': 'accuracy'})

    return model

def format_metrics(metrics, model):
    metrics = [round(float(m), 8) for m in metrics]
    metrics_dict = dict(zip(model.metrics_names, metrics))
    return json.dumps(metrics_dict)

def train(model):
    train_set = get_session_dicts()

    # Score must be at least 10 of 20
    train_set = list(filter(lambda s: s['score'] >= 10, train_set))
    test_set = split_off_test_set(train_set)

    # Experiment: Overfit single batch
    #train_set = train_set[:BATCH_SIZE]
    #for t in train_set:
    #    t['volume'] = SEQUENCE_LENGTH

    train_set, total_train_batches = split_into_super_batches(train_set)
    test_set, total_test_batches = split_into_super_batches(test_set)

    for epoch in range(EPOCHS):
        for batch, x, y, mask in state_control(model, iterate_batches(train_set)):
            metrics = model.train_on_batch(x, y, mask, reset_metrics=False)
            print("Batch {}/{} {}".format(batch+1, total_train_batches, format_metrics(metrics, model)))

        print("Epoch {}/{} training metrics {}".format(epoch+1, EPOCHS, format_metrics(metrics, model)))
        model.reset_metrics()

        save_filename = os.path.join(CHECKPOINT_DIR, "cp-{:04}.ckpt".format(epoch))
        print("Saving model to", save_filename)
        model.save_weights(save_filename)

        for batch, x, y, mask in state_control(model, iterate_batches(test_set)):
            metrics = model.test_on_batch(x, y, mask, reset_metrics=False)
        print("Epoch {}/{} validation metrics {}".format(epoch+1, EPOCHS, format_metrics(metrics, model)))
        model.reset_metrics()

def time_forward_pass():
    import time

    model = build_model(live=True)
    x = np.random.randint(0, TILE_COUNT, (1, 1, 50, 90))

    model.predict_on_batch(x) # First prediction takes longer

    before = time.time()
    for i in range(100):
        model.predict_on_batch(x)
    seconds = (time.time() - before) / 100

    print("Forward pass takes {:.5f} seconds".format(seconds))

    model.reset_states()

def estimate_model_memory_usage(model):
    from tensorflow.keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in (l.output_shape[0] if isinstance(l.output_shape, list) else l.output_shape):
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

def already_exists_bug_workaround():
    from tensorflow.core.protobuf import rewriter_config_pb2
    from tensorflow.keras.backend import set_session
    #tf.keras.backend.clear_session() # For easy reset of notebook state.

    config_proto = tf.ConfigProto()
    off = rewriter_config_pb2.RewriterConfig.OFF
    config_proto.graph_options.rewrite_options.arithmetic_optimization = off
    session = tf.Session(config=config_proto)
    set_session(session)


if __name__ == '__main__':
    #np.random.seed(1)
    #tf.random.set_random_seed(1)

    model = build_model()
    #model.summary(100)
    #print(estimate_model_memory_usage(model), 'GiB')

    #already_exists_bug_workaround() # Also try downgrading to 1.12
    train(model)

    #time_forward_pass()
