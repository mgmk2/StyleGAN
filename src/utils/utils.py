import tensorflow as tf

def num_div2(x):
    n = 0
    while(x % 2 == 0):
        x = x // 2
        n += 1
    return n

def convert_to_tensor(inputs, dtype=None):
    if dtype is None:
        dtype = tf.float32
    if isinstance(inputs, list) or isinstance(inputs, tuple):
        outputs = [None for _ in range(len(inputs))]
        for i, x in enumerate(inputs):
            outputs[i] = tf.convert_to_tensor(x, dtype=dtype)
        return tuple(outputs)
    return tf.convert_to_tensor(inputs, dtype=dtype)

def assert_all_finite(*inputs):
    for x in inputs:
        message = x.name + ' has non-numeric value.'
        tf.debugging.assert_all_finite(x, message)
