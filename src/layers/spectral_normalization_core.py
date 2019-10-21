# Code derived from
# https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/layers/core.py
# and
# https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/layers/normalization.py

import numpy as np

from functools import reduce
from operator import mul

from tensorflow.python.distribute import distribution_strategy_context

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
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
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops.custom_gradient import custom_gradient

from tensorflow.python.keras.layers.core import Dense

class SNDense(Dense):
    """Just your regular densely-connected NN layer with spectral normalization.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: If the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.

    Example:

    ```python
    # as first layer in a sequential model:
    model = Sequential()
    model.add(Dense(32, input_shape=(16,)))
    # now the model will take as input arrays of shape (*, 16)
    # and output arrays of shape (*, 32)

    # after the first layer, you don't need to specify
    # the size of the input anymore:
    model.add(Dense(32))
    ```

    Arguments:
    units: Positive integer, dimensionality of the output space.
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
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    singular_vector_initializer: Initializer for
        the singular vector used in spectral normalization.
    power_iter: Positive integer,
        number of iteration in singular value estimation.

    Input shape:
    N-D tensor with shape: `(batch_size, ..., input_dim)`.
    The most common situation would be
    a 2D input with shape `(batch_size, input_dim)`.

    Output shape:
    N-D tensor with shape: `(batch_size, ..., units)`.
    For instance, for a 2D input with shape `(batch_size, input_dim)`,
    the output would have shape `(batch_size, units)`.
    """

    def __init__(self,
                 units,
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
                 **kwargs):

        super(SNDense, self).__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
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
        if self.use_wscale:
            fan_in = int(input_shape[-1])
            self.coeff = np.sqrt(2 / fan_in)
        else:
            self.coeff = 1.0

        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point '
                            'dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        last_dim = tensor_shape.dimension_value(input_shape[-1])

        self.u = self.add_weight(
            name='singular_vector',
            shape=(1, last_dim),
            initializer=self.singular_vector_initializer,
            trainable=False,
            aggregation=tf_variables.VariableAggregation.ONLY_FIRST_REPLICA,
            dtype=self.dtype)
        super(SNDense, self).build(input_shape)

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
            W = self.coeff * self.kernel
        else:
            @custom_gradient
            def lr_multiplier(x):
                y = array_ops.identity(x)
                def grad(dy):
                    return dy * self.lr_mul
                return y, grad
            W = lr_multiplier(self.coeff * self.kernel)

        training = self._get_training_value(training)

        # Update singular vector by power iteration
        W_T = array_ops.transpose(W)
        u = array_ops.identity(self.u)
        for i in range(self.power_iter):
            v = nn_impl.l2_normalize(math_ops.matmul(u, W))  # 1 x filters
            u = nn_impl.l2_normalize(math_ops.matmul(v, W_T))
        # Spectral Normalization
        sigma_W = math_ops.matmul(math_ops.matmul(u, W), array_ops.transpose(v))
        # Backprop doesn't need in power iteration
        sigma_W = array_ops.stop_gradient(sigma_W)
        W_bar = W / array_ops.squeeze(sigma_W)

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

        # normal Dense using W_bar
        inputs = ops.convert_to_tensor(inputs)
        rank = common_shapes.rank(inputs)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, W_bar, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            inputs = math_ops.cast(inputs, self._compute_dtype)
            outputs = math_ops.mat_mul(inputs, W_bar)
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def get_config(self):
        config = {
            'singular_vector_initializer': initializers.serialize(
                                           self.singular_vector_initializer),
            'power_iter': self.power_iter
        }
        base_config = super(SNDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
