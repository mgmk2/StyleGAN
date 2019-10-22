from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.python.keras.engine.base_layer import Layer

from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops

class ScaleAddToConst(Layer):
    """ This layer returns the sum of the scaled input and the constant variable.
    The constant variable is 4D tensor, (1, rows, cols, channels).
    Usually this layer is used as Input layer of 2D convolution in StyleGAN.

    Arguments:
    const_shape: tuple/list of n integers,
        the shape of the constant variable without batch size.
    const_initializer: An initializer for the constant variable.
    scale_initializer: An initializer for the scale weights.
    trainable: Boolean, if `True` the weights of this layer will be marked as
      trainable (and listed in `layer.trainable_weights`).
    name: A string, the name of the layer.

    Input shape:
    3D tensor with shape: `(batch_size, rows, cols)`
    or 4D tensor with shape: `(batch_size, rows, cols, 1)`.
    Rows and cols must be equal to ones of the constant variable.

    Output shape:
    4D tensor with shape: `(batch_size, rows, cols, channels)`.
    """

    def __init__(self, const_shape,
                 const_initializer='ones',
                 scale_initializer='zeros',
                 trainable=True,
                 name=None,
                 **kwargs):
        super(ScaleAddToConst, self).__init__(
            trainable=trainable,
            name=name,
            **kwargs)
        self.const_shape = (1,) + const_shape
        self.const_initializer = initializers.get(const_initializer)
        self.scale_initializer = initializers.get(scale_initializer)

    def build(self, input_shape):
        assert input_shape[1] == self.const_shape[1]
        assert input_shape[2] == self.const_shape[2]
        self.scale = self.add_weight(
            name='scale',
            shape=(1, 1, 1, self.const_shape[-1]),
            initializer=self.scale_initializer,
            trainable=True,
            dtype=self.dtype)
        self.const = self.add_weight(
            name='const',
            shape=self.const_shape,
            initializer=self.const_initializer,
            trainable=True,
            dtype=self.dtype)
        self.build = True

    def call(self, inputs):
        if len(inputs.shape) == 3:
            inputs = K.expand_dims(inputs, axis=-1)
        return self.const + self.scale * inputs

class ScaleAdd(Layer):
    """ This layer returns the sum of 1st input and scaled 2nd input.
    Usually this layer is used as middle layer in StyleGAN.

    Arguments:
    scale_initializer: An initializer for the scale weights.
    trainable: Boolean, if `True` the weights of this layer will be marked as
      trainable (and listed in `layer.trainable_weights`).
    name: A string, the name of the layer.

    Input shape:
    tuple/list of 2 tensors.
    1st tensor: 4D tensor with shape: `(batch_size, rows, cols, channels)`.
    2nd tensor: 3D tensor with shape: `(batch_size, rows, cols)`
        or 4D tensor with shape: `(batch_size, rows, cols, 1)`.

    Output shape:
    4D tensor with shape: `(batch_size, rows, cols, channels)`.
    """

    def __init__(self, scale_initializer='zeros',
                 trainable=True,
                 name=None,
                 **kwargs):
        super(ScaleAdd, self).__init__(
            trainable=trainable,
            name=name,
            **kwargs)
        self.scale_initializer = initializers.get(scale_initializer)

    def build(self, input_shape):
        if not isinstance(input_shape, list) and not isinstance(input_shape, tuple):
            raise ValueError('A ScaleAdd layer should be called '
                             'on a list/tuple of inputs.')
        if len(input_shape) < 2:
            raise ValueError('A ScaleAdd layer should be called '
                             'on a list/tuple of 2 inputs. '
                             'Got ' + str(len(input_shape)) + ' inputs.')
        input1_shape, input2_shape = input_shape
        self.scale = self.add_weight(
            name='scale',
            shape=(1, 1, 1, input1_shape[-1]),
            initializer=self.scale_initializer,
            trainable=True,
            dtype=self.dtype)
        self.build = True

    def call(self, inputs):
        if not isinstance(inputs, list) and not isinstance(inputs, tuple):
            raise ValueError('A ScaleAdd layer should be called '
                             'on a list/tuple of inputs.')
        if len(inputs) != 2:
            raise ValueError('A ScaleAdd layer should be called '
                             'on a list/tuple of 2 inputs. '
                             'Got ' + str(len(inputs)) + ' inputs.')
        x1, x2 = inputs
        if len(x2.shape) == 3:
            x2 = K.expand_dims(x2, axis=-1)
        return x1 + self.scale * x2

class AdaIN(Layer):
    """ AdaIN layer. Usually used as middle layer in StyleGAN.

    Arguments:
    epsilon: Epsilon value for division.
    trainable: Boolean, if `True` the weights of this layer will be marked as
      trainable (and listed in `layer.trainable_weights`).
    name: A string, the name of the layer.

    Input shape:
    tuple/list of 2 tensors: `(x, weight)`.
    x: N-D tensor with shape: `(batch_size, ...)`.
    weight: (N + 1)-D tensor with shape: `(batch_size, 2, ...)`.

    Output shape:
    N-D tensor with same shape as x.
    """

    def __init__(self, epsilon=1.0e-8, name=None, **kwargs):
        super(AdaIN, self).__init__(name=name, **kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        if not isinstance(input_shape, list) and not isinstance(input_shape, tuple):
            raise ValueError('An AdaIN layer should be called '
                             'on a list/tuple of inputs.')
        if len(input_shape) != 2:
            raise ValueError('An AdaIN layer should be called '
                             'on a list/tuple of 2 inputs. '
                             'Got ' + str(len(input_shape)) + ' inputs.')
        x_shape, w_shape = input_shape
        assert len(w_shape) == len(x_shape) + 1
        assert w_shape[1] == 2
        self.build =True

    def _instance_normalize(self, x):
        x_dtype = x.dtype
        x -= math_ops.reduce_mean(x, axis=[1, 2], keepdims=True)
        epsilon = K.constant(self.epsilon, dtype=x_dtype, name='epsilon')
        x *= math_ops.rsqrt(
            math_ops.reduce_mean(x ** 2, axis=[1, 2], keepdims=True) + epsilon)
        return x

    def call(self, inputs):
        if not isinstance(inputs, list) and not isinstance(inputs, tuple):
            raise ValueError('An AdaIN layer should be called '
                             'on a list/tuple of inputs.')
        if len(inputs) != 2:
            raise ValueError('An AdaIN layer should be called '
                             'on a list/tuple of 2 inputs. '
                             'Got ' + str(len(inputs)) + ' inputs.')
        x, w = inputs
        w_s = w[:, 0]
        w_b = w[:, 1]
        return w_s * self._instance_normalize(x) + w_b

class AddBias2D(Layer):
    """ Add bias layer for 4D tensor.
    Same as adding bias process of Conv2D layer.

    Arguments:
    bias_initializer: An initializer for the bias vector. If None, the default
        initializer will be used.
    bias_regularizer: Optional regularizer for the bias vector.
    bias_constraint: Optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
    trainable: Boolean, if `True` the weights of this layer will be marked as
      trainable (and listed in `layer.trainable_weights`).
    name: A string, the name of the layer.

    Input shape:
    4D tensor with shape: `(batch_size, rows, cols, channels)`.

    Output shape:
    4D tensor with same shape as input.
    """

    def __init__(self,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 name=None,
                 **kwargs):
        super(AddBias2D, self).__init__(name=name, **kwargs)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError('An AddBias2D layer should be called '
                             'on a 4D Tensor. '
                             'Got ' + str(len(input_shape)) + 'D Tensor.')
        self.bias = self.add_weight(
            name='bias',
            shape=(input_shape[-1],),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            trainable=True,
            dtype=self.dtype)
        self.build = True

    def call(self, inputs):
        if len(inputs.shape) != 4:
            raise ValueError('An AddBias2D layer should be called '
                             'on a 4D Tensor. '
                             'Got ' + str(len(inputs.shape)) + 'D Tensor.')
        outputs = nn.bias_add(inputs, self.bias, data_format='NHWC')
        return outputs
