# Code derived from
# https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/layers/core.py
# and
# https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/layers/normalization.py

import numpy as np

from tensorflow.python.distribute import distribution_strategy_context

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables

class MixStyle(Layer):
    def __init__(self, num_layers,
                 mixing_prob=0.9,
                 latent_avg_beta=0.995,
                 truncation_psi=0.7,
                 truncation_cutoff=8,
                 name=None,
                 **kwargs):
        super(MixStyle, self).__init__(name=name, **kwargs)
        self.num_layers = num_layers
        self.mixing_prob = mixing_prob
        self.latent_avg_beta = latent_avg_beta
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff
        self.layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]

        self.update_latent_avg = (self.latent_avg_beta is not None)
        self.mix_latents = (self.mixing_prob is not None)
        self.truncate_latent = (self.truncation_psi is not None and
                                self.truncation_cutoff is not None)

        if self.truncation_psi is not None and self.truncation_cutoff is not None:
            ones = array_ops.ones([1, num_layers, 1], dtype=dtypes.float32)
            self.coeff = array_ops.where(
                self.layer_idx < self.truncation_cutoff,
                self.truncation_psi * ones,
                ones)
        self._trainable_var = None

    def build(self, input_shape):
        latent1_shape, latent2_shape, lod_shape = input_shape
        self.latent_avg = self.add_weight(
            'latent_avg',
            shape=(latent1_shape[-1],),
            initializer=initializers.get('zeros'),
            trainable=False,
            aggregation=tf_variables.VariableAggregation.MEAN,
            dtype=self.dtype)
        self.build = True

    def _assign_latent_avg(self, variable, value):
        with K.name_scope('latent_avg') as scope:
            with ops.colocate_with(variable):
                return state_ops.assign(variable, value, name=scope)

    def _get_trainable_var(self):
        if self._trainable_var is None:
            self._trainable_var = K.freezable_variable(
                self._trainable, name=self.name + '_trainable')
        return self._trainable_var

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

    def _interpolate(self, x1, x2, ratio):
        return x1 + ratio * (x2 - x1)

    def call(self, inputs, training=None):
        training = self._get_training_value(training)
        latent1, latent2, lod = inputs

        training_value = tf_utils.constant_value(training)
        latent_avg_new = math_ops.reduce_mean(latent1[:, 0], axis=0)
        if training_value != False and self.update_latent_avg:
            latent_avg_new = self._interpolate(
                latent_avg_new, self.latent_avg, self.latent_avg_beta)

            def update_op():
                def true_branch():
                    return self._assign_latent_avg(self.latent_avg, latent_avg_new)
                def false_branch():
                    return self.latent_avg
                return tf_utils.smart_cond(training, true_branch, false_branch)
            self.add_update(update_op)

        if training_value != False and self.mix_latents:
            def true_branch():
                cur_layer = 2 * (1 + math_ops.cast(
                    array_ops.reshape(lod, [-1])[0], dtypes.int32))
                cutoff = tf_utils.smart_cond(
                    random_ops.random_uniform([], 0.0, 1.0) < self.mixing_prob,
                    lambda: random_ops.random_uniform([], 1, cur_layer, dtypes.int32),
                    lambda: cur_layer)
                return array_ops.where(
                    array_ops.broadcast_to(
                        self.layer_idx < cutoff, array_ops.shape(latent1)),
                    latent1,
                    latent2)
            def false_branch():
                return latent1
            latent1 = tf_utils.smart_cond(training, true_branch, false_branch)

        if training_value != True and self.truncate_latent:
            def true_branch():
                return self._interpolate(latent_avg_new, latent1, self.coeff)
            def false_branch():
                return latent1
            latent1 = tf_utils.smart_cond(training, true_branch, false_branch)

        return latent1
