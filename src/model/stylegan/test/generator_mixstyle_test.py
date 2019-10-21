import numpy as np
import tensorflow as tf
from src.model.stylegan.network import build_style_mixer

if __name__ == '__main__':

    batch_size = 16

    lod = tf.placeholder(tf.float32, [None, 1])
    latent1 = tf.placeholder(tf.float32, [None, 8, 512])
    latent2 = tf.placeholder(tf.float32, [None, 8, 512])

    model = build_style_mixer()
    outputs = model([lod, latent1, latent2], training=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y = sess.run(outputs,
            feed_dict={lod: 4 * np.ones([batch_size, 1], dtype=np.float32),
                      latent1: np.ones([batch_size, 8, 512], dtype=np.float32),
                      latent2: np.zeros([batch_size, 8, 512], dtype=np.float32)})
        print(y.shape)
