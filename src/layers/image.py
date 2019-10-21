import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine.base_layer import Layer

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops

class Blur(Layer):
    def __init__(self, filter=(1, 2, 1),
                 normalize=True,
                 stride=1,
                 name=None,
                 **kwargs):
        super(Blur, self).__init__(name=name, **kwargs)
        self.filter = filter
        self.normalize = normalize
        self.stride = stride

    def call(self, inputs):
        filter = np.array(self.filter, np.float32)
        if filter.ndim == 1:
            filter = filter[:, np.newaxis] * filter[np.newaxis, :]
        if self.normalize:
            filter /= np.sum(filter)
        filter = filter[:, :, np.newaxis, np.newaxis]
        filter = K.constant(filter, dtype=inputs.dtype, name='filter')
        filter = K.tile(filter, [1, 1, K.shape(inputs)[-1], 1])
        outputs = nn.depthwise_conv2d(
            inputs,
            filter,
            strides=(1, self.stride, self.stride, 1),
            padding='SAME')
        return outputs

class UpSampling2D(Layer):
    def __init__(self, factor=(2, 2), name=None, **kwargs):
        super(UpSampling2D, self).__init__(name=name, **kwargs)
        self.factor = factor

    def call(self, inputs):
        shape = inputs.shape
        r1, r2 = self.factor
        h = array_ops.reshape(
            inputs, [-1, shape[1], 1, shape[2], 1, shape[3]])
        h = array_ops.tile(h, [1, 1, r1, 1, r2, 1])
        y = array_ops.reshape(h, [-1, r1 * shape[1], r2 * shape[2], shape[3]])
        return y
