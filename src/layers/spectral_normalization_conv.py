# Code derived from
# https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/layers/convolutional.py
# and
# https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/layers/normalization.py

import numpy as np

from functools import reduce
from operator import mul

from tensorflow.python.distribute import distribution_strategy_context

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape

from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers

from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.custom_gradient import custom_gradient

from tensorflow.python.keras.layers.convolutional import Conv

class SNConv(Conv):
    """Abstract N-D convolution layer with spectral normalization
        (private, used as implementation base).

    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True (and a `bias_initializer` is provided),
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    Arguments:
    rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
    filters: Integer, the dimensionality of the output space (i.e. the number
      of filters in the convolution).
    kernel_size: An integer or tuple/list of n integers, specifying the
      length of the convolution window.
    strides: An integer or tuple/list of n integers,
      specifying the stride length of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"`,  `"same"`, or `"causal"` (case-insensitive).
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, ..., channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, ...)`.
    dilation_rate: An integer or tuple/list of n integers, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    activation: Activation function. Set it to None to maintain a
      linear activation.
    use_bias: Boolean, whether the layer uses a bias.
    kernel_initializer: An initializer for the convolution kernel.
    bias_initializer: An initializer for the bias vector. If None, the default
      initializer will be used.
    kernel_regularizer: Optional regularizer for the convolution kernel.
    bias_regularizer: Optional regularizer for the bias vector.
    activity_regularizer: Optional regularizer function for the output.
    kernel_constraint: Optional projection function to be applied to the
        kernel after being updated by an `Optimizer` (e.g. used to implement
        norm constraints or value constraints for layer weights). The function
        must take as input the unprojected variable and must return the
        projected variable (which must have the same shape). Constraints are
        not safe to use when doing asynchronous distributed training.
    bias_constraint: Optional projection function to be applied to the
        bias after being updated by an `Optimizer`.
    singular_vector_initializer: Initializer for
        the singular vector used in spectral normalization.
    power_iter: Positive integer,
        number of iteration in singular value estimation.
    trainable: Boolean, if `True` the weights of this layer will be marked as
      trainable (and listed in `layer.trainable_weights`).
    name: A string, the name of the layer.
    """

    def __init__(self, rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 use_wscale=False,
                 lr_mul=1.0,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 singular_vector_initializer=initializers.RandomNormal(0, 1),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 power_iter=1,
                 trainable=True,
                 name=None,
                 **kwargs):

        super(SNConv, self).__init__(
            rank=rank,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            name=name,
            **kwargs)
        self.use_wscale = use_wscale
        self.lr_mul = lr_mul
        self.singular_vector_initializer = singular_vector_initializer
        self.power_iter = power_iter
        self._trainable_var = None
        self.trainable = trainable

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value
        if self._trainable_var is not None:
            self._trainable_var.update_value(value)

    def _get_trainable_var(self):
        if self._trainable_var is None:
            self._trainable_var = K.freezable_variable(
                self._trainable, name=self.name + '_trainable')
        return self._trainable_var

    def build(self, input_shape):
        super(SNConv, self).build(input_shape)
        if self.use_wscale:
            fan_in = reduce(mul, self.kernel_size) * int(input_shape[-1])
            self.coeff = np.sqrt(2 / fan_in)
        else:
            self.coeff = 1.0

        input_shape = tensor_shape.TensorShape(input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        singular_vector_shape = (1, reduce(mul, self.kernel_size) * input_dim)

        self.u = self.add_weight(
            name='singular_vector',
            shape=singular_vector_shape,
            initializer=self.singular_vector_initializer,
            trainable=False,
            aggregation=tf_variables.VariableAggregation.ONLY_FIRST_REPLICA,
            dtype=self.dtype)

    def _get_training_value(self, training=None):
        if training is None:
            training = K.learning_phase()
        if isinstance(training, int):
            training = bool(training)
        if base_layer_utils.is_in_keras_graph():
            training = math_ops.logical_and(training, self._get_trainable_var())
        else:
            training = math_ops.logical_and(training, self.trainable)
        return training

    def _assign_singular_vector(self, variable, value):
        with K.name_scope('AssignSingularVector') as scope:
            with ops.colocate_with(variable):
                return state_ops.assign(variable, value, name=scope)

    def call(self, inputs, training=None):
        if self.lr_mul == 1.0:
            kernel = self.coeff * self.kernel
        else:
            @custom_gradient
            def lr_multiplier(x):
                y = array_ops.identity(x)
                def grad(dy):
                    return dy * self.lr_mul
                return y, grad
            kernel = lr_multiplier(self.coeff * self.kernel)

        training = self._get_training_value(training)

        # Update singular vector by power iteration
        if self.data_format == 'channels_first':
            W_T = array_ops.reshape(kernel, (self.filters, -1))
            W = array_ops.transpose(W_T)
        else:
            W = array_ops.reshape(kernel, (-1, self.filters))
            W_T = array_ops.transpose(W)
        u = array_ops.identity(self.u)
        for i in range(self.power_iter):
            v = nn_impl.l2_normalize(math_ops.matmul(u, W))  # 1 x filters
            u = nn_impl.l2_normalize(math_ops.matmul(v, W_T))
        # Spectral Normalization
        sigma_W = math_ops.matmul(math_ops.matmul(u, W), array_ops.transpose(v))
        # Backprop doesn't need in power iteration
        sigma_W = array_ops.stop_gradient(sigma_W)
        W_bar = kernel / array_ops.squeeze(sigma_W)

        # Assign new singular vector
        training_value = tf_utils.constant_value(training)
        if training_value is not False:
            def u_update():
                def true_branch():
                    return self._assign_singular_vector(self.u, u)
                def false_branch():
                    return self.u
                return tf_utils.smart_cond(training, true_branch, false_branch)
            self.add_update(u_update)

        # normal convolution using W_bar
        outputs = self._convolution_op(inputs, W_bar)

        if self.use_bias:
          if self.data_format == 'channels_first':
            if self.rank == 1:
              # nn.bias_add does not accept a 1D input tensor.
              bias = array_ops.reshape(self.bias, (1, self.filters, 1))
              outputs += bias
            else:
              outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
          else:
            outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
          return self.activation(outputs)
        return outputs

    def get_config(self):
        config = {
            'singular_vector_initializer': initializers.serialize(
                                           self.singular_vector_initializer),
            'power_iter': self.power_iter
        }
        base_config = super(SNConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SNConv1D(SNConv):
    """1D convolution layer with spectral normalization
        (e.g. temporal convolution).

    This layer creates a convolution kernel that is convolved
    with the layer input over a single spatial (or temporal) dimension
    to produce a tensor of outputs.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.

    When using this layer as the first layer in a model,
    provide an `input_shape` argument
    (tuple of integers or `None`, e.g.
    `(10, 128)` for sequences of 10 vectors of 128-dimensional vectors,
    or `(None, 128)` for variable-length sequences of 128-dimensional vectors.

    Arguments:
    filters: Integer, the dimensionality of the output space
      (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of a single integer,
      specifying the length of the 1D convolution window.
    strides: An integer or tuple/list of a single integer,
      specifying the stride length of the convolution.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: One of `"valid"`, `"causal"` or `"same"` (case-insensitive).
      `"causal"` results in causal (dilated) convolutions, e.g. output[t]
      does not depend on input[t+1:]. Useful when modeling temporal data
      where the model should not violate the temporal order.
      See [WaveNet: A Generative Model for Raw Audio, section
        2.1](https://arxiv.org/abs/1609.03499).
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
    dilation_rate: an integer or tuple/list of a single integer, specifying
      the dilation rate to use for dilated convolution.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any `strides` value != 1.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation")..
    kernel_constraint: Constraint function applied to the kernel matrix.
    bias_constraint: Constraint function applied to the bias vector.
    singular_vector_initializer: Initializer for
        the singular vector used in spectral normalization.
    power_iter: Positive integer,
        number of iteration in singular value estimation.

    Input shape:
    3D tensor with shape: `(batch_size, steps, input_dim)`

    Output shape:
    3D tensor with shape: `(batch_size, new_steps, filters)`
      `steps` value might have changed due to padding or strides.
    """

    def __init__(self,
               filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               use_wscale=False,
               lr_mul=1.0,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               singular_vector_initializer=initializers.RandomNormal(0, 1),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               power_iter=1,
               **kwargs):
        super(SNConv1D, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            use_wscale=use_wscale,
            lr_mul=lr_mul,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            singular_vector_initializer=initializers.get(singular_vector_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            power_iter=power_iter,
            **kwargs)

    def call(self, inputs):
        if self.padding == 'causal':
          inputs = array_ops.pad(inputs, self._compute_causal_padding())
        return super(SNConv1D, self).call(inputs)


class SNConv2D(SNConv):
    """2D convolution layer with spectral normalization
        (e.g. spatial convolution over images).

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is True,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.

    Arguments:
    filters: Integer, the dimensionality of the output space
      (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the
      height and width of the 2D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the convolution along the height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch, channels, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
    dilation_rate: an integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation")..
    kernel_constraint: Constraint function applied to the kernel matrix.
    bias_constraint: Constraint function applied to the bias vector.
    singular_vector_initializer: Initializer for
        the singular vector used in spectral normalization.
    power_iter: Positive integer,
        number of iteration in singular value estimation.

    Input shape:
    4D tensor with shape:
    `(samples, channels, rows, cols)` if data_format='channels_first'
    or 4D tensor with shape:
    `(samples, rows, cols, channels)` if data_format='channels_last'.

    Output shape:
    4D tensor with shape:
    `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
    or 4D tensor with shape:
    `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
    `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 use_wscale=False,
                 lr_mul=1.0,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 singular_vector_initializer=initializers.RandomNormal(0, 1),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 power_iter=1,
                 **kwargs):
        super(SNConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            use_wscale=use_wscale,
            lr_mul=lr_mul,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            singular_vector_initializer=initializers.get(singular_vector_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            power_iter=power_iter,
            **kwargs)


class SNConv3D(SNConv):
    """3D convolution layer with spectral Normalization
        (e.g. spatial convolution over volumes).

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is True,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 128, 1)` for 128x128x128 volumes
    with a single channel,
    in `data_format="channels_last"`.

    Arguments:
    filters: Integer, the dimensionality of the output space
      (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of 3 integers, specifying the
      depth, height and width of the 3D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 3 integers,
      specifying the strides of the convolution along each spatial
        dimension.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      while `channels_first` corresponds to inputs with shape
      `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
    dilation_rate: an integer or tuple/list of 3 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation")..
    kernel_constraint: Constraint function applied to the kernel matrix.
    bias_constraint: Constraint function applied to the bias vector.
    singular_vector_initializer: Initializer for
        the singular vector used in spectral normalization.
    power_iter: Positive integer,
        number of iteration in singular value estimation.

    Input shape:
    5D tensor with shape:
    `(samples, channels, conv_dim1, conv_dim2, conv_dim3)` if
      data_format='channels_first'
    or 5D tensor with shape:
    `(samples, conv_dim1, conv_dim2, conv_dim3, channels)` if
      data_format='channels_last'.

    Output shape:
    5D tensor with shape:
    `(samples, filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)` if
      data_format='channels_first'
    or 5D tensor with shape:
    `(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, filters)` if
      data_format='channels_last'.
    `new_conv_dim1`, `new_conv_dim2` and `new_conv_dim3` values might have
      changed due to padding.
    """

    def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1, 1),
               padding='valid',
               data_format=None,
               dilation_rate=(1, 1, 1),
               activation=None,
               use_bias=True,
               use_wscale=False,
               lr_mul=1.0,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               singular_vector_initializer=initializers.RandomNormal(0, 1),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               power_iter=1,
               **kwargs):
        super(SNConv3D, self).__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            use_wscale=use_wscale,
            lr_mul=lr_mul,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            singular_vector_initializer=initializers.get(singular_vector_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            power_iter=power_iter,
            **kwargs)
