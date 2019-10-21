from tensorflow.python.keras.engine.base_layer import Layer

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

class PixelNormalization(Layer):
    def __init__(self, epsilon=1.0e-8, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs):
        var = math_ops.reduce_mean(
            math_ops.square(inputs), axis=-1, keepdims=True)
        return inputs * math_ops.rsqrt(var + self.epsilon)


class BatchStddev(Layer):
    def __init__(self, group_size=4, num_features=1, **kwargs):
        super(BatchStddev, self).__init__(**kwargs)
        self.group_size = group_size
        self.num_features = num_features

    def build(self, input_shape):
        self.shape = input_shape
        self.build =True

    def call(self, inputs):
        if inputs.shape[0] is not None:
            self.group_size = min(self.group_size, inputs.shape[0])
        shape = (self.group_size, -1, self.shape[1], self.shape[2],
            self.shape[3] // self.num_features, self.num_features)
        x = array_ops.reshape(inputs, shape)
        x -= math_ops.reduce_mean(x, axis=0, keepdims=True)
        x = math_ops.reduce_mean(math_ops.square(x), axis=0)
        x = math_ops.sqrt(x)
        x = math_ops.reduce_mean(x, axis=[1, 2], keepdims=True)
        x = math_ops.reduce_mean(x, axis=3)
        x = array_ops.tile(
            x, (self.group_size, self.shape[1], self.shape[2], 1))
        return array_ops.concat([inputs, x], axis=-1)
