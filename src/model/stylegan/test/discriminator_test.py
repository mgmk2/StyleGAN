import numpy as np
import tensorflow as tf
from src.model.stylegan.network import build_discriminator

if __name__ == '__main__':

    batch_size = 16

    lod = tf.placeholder(tf.float32)
    image = tf.placeholder(tf.float32, [None, 8, 8, 3])

    D = build_discriminator(res=8, num_channels_in=3, mode='dynamic')
    outputs = D([lod, image])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y = sess.run(outputs,
            feed_dict={lod: 0,
                       image: np.ones([batch_size, 8, 8, 3], dtype=np.float32)})
        print(y.shape)
