#!/usr/bin/python3
from a2c import tf, A2CNetwork, config_proto

with tf.Session(config=config_proto) as sess:
    # Load latest checkpoint
    net = A2CNetwork(sess)
    episode = net.load_variables()

    # Write as SavedModel
    net.write_model_to_disk()
