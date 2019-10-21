import numpy as np
import tensorflow as tf
from src.model.stylegan.network import build_generator_mapping

if __name__ == '__main__':

    batch_size = 16

    lod = tf.placeholder(tf.float32, [1, 1])
    z = tf.placeholder(tf.float32, [None, 512])

    model = build_generator_mapping()
    outputs = model(z, training=True)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y = sess.run(outputs,
            feed_dict={z: np.zeros([batch_size, 512], dtype=np.float32)})
        print(y.shape)
