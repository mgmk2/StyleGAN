import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from scale_add import ScaleAddToConst

inputs = tf.placeholder(tf.float32, [None, 4, 4, 1])

z = Input((4, 4, 1))
y = ConstBlock((4, 4, 512))(z)
model = Model(z, y)

outputs = model(inputs)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    noise = np.ones([1, 4, 4, 1], dtype=np.float32)
    print(sess.run(outputs, feed_dict={inputs: noise}))
