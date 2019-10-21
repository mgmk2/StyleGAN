import numpy as np
import tensorflow as tf
from src.model.stylegan.network import build_generator_synthesis

if __name__ == '__main__':

    batch_size = 16

    lod = tf.placeholder(tf.float32)
    w = tf.placeholder(tf.float32, [None, 6, 64])
    noise0 = tf.placeholder(tf.float32, [None, 4, 4, 2])
    noise1 = tf.placeholder(tf.float32, [None, 8, 8, 2])
    noise2 = tf.placeholder(tf.float32, [None, 16, 16, 2])

    generator = build_generator_synthesis(
        res_out=16, num_channels=3, num_latent=64, mode='dynamic')
    outputs = generator([lod, w, noise0, noise1, noise2])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y = sess.run(outputs,
            feed_dict={lod: 1,
                      w: np.ones([batch_size, 6, 64], dtype=np.float32),
                      noise0: np.zeros([batch_size, 4, 4, 2], dtype=np.float32),
                      noise1: np.zeros([batch_size, 8, 8, 2], dtype=np.float32),
                      noise2: np.zeros([batch_size, 16, 16, 2], dtype=np.float32)})
        print(y.shape)
