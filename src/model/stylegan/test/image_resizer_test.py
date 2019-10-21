import numpy as np
import tensorflow as tf
from src.model.stylegan.network import image_resizer

res = 32

inputs = tf.placeholder(tf.float32, [None, res, res, 3])
lod = tf.placeholder(tf.float32)
outputs = image_resizer(inputs, lod, res, mode='linear')

with tf.Session() as sess:
    x = np.arange(res * res).reshape([1, res, res, 1]).astype(np.float32)
    x = np.tile(x, [1, 1, 1, 3])
    sess.run(tf.global_variables_initializer())
    y = sess.run(outputs, feed_dict={inputs: x, lod: 0})
    print(y)
