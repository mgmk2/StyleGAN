import numpy as np
import tensorflow as tf
from .network import GeneratorMapping, GeneratorSynthesis, \
                     Discriminator, StyleMixer, res2num_blocks, \
                     image_resizer
from ..base_model import BaseModel
from ...utils.decorator import tpu_decorator, tpu_ops_decorator, convert_to_tfdata_single_batch

class StyleGANModel(BaseModel):
    def __init__(self, params, use_tpu=False, mode=None):
        super().__init__(params, use_tpu=use_tpu)
        self.z_dim = self.params.z_dim
        self.num_mapping_layers = getattr(params, 'num_layers', 8)
        assert self.params.image_shape[0] == self.params.image_shape[1]
        self.image_res = self.params.image_shape[0]
        self.image_ch = self.params.image_shape[-1]

        self.use_wscale = getattr(params, 'use_wscale', True)
        self.lr_mul = getattr(
            params, 'lr_mul',
            {'gen_mapping': 1.0, 'gen_synthesis': 1.0, 'disc': 1.0})

        self.mixing_prob = getattr(params, 'mixing_prob', 0.9)
        self.latent_avg_beta = getattr(params, 'latent_avg_beta', 0.995)
        self.truncation_psi = getattr(params, 'truncation_psi', 0.7)
        self.truncation_cutoff = getattr(params, 'truncation_cutoff', 8)
        self.batch_std_group_size = getattr(params, 'batch_std_group_size', 4)
        self.batch_std_num_features = getattr(params, 'batch_std_num_features', 1)

        self.use_sn_in_disc = getattr(params, 'use_sn_in_disc', False)

        if self.use_tpu:
            if mode != 'static':
                print(('Warning! StyleGAN only supports static mode on TPU. '
                       'Change mode to static automatically.'))
            self.mode = 'static'
        else:
            self.mode = 'dynamic' if mode is None else mode

        self.build_model()

    def get_learning_rate(self):
        if hasattr(self.params, 'lr_schedule'):
            return tf.optimizers.schedules.PiecewiseConstantDecay(
                self.params.lr_schedule['boundaries'],
                self.params.lr_schedule['values'])
        return self.params.learning_rate

    @tpu_decorator
    def build_model(self):

        self.discriminator = Discriminator(
            res=self.image_res,
            num_channels_in=self.image_ch,
            num_channels=1,
            fmap_base=8192,
            fmap_decay=1.0,
            fmap_max=self.z_dim,
            mode=self.mode,
            use_wscale=self.use_wscale,
            lr_mul=self.lr_mul['disc'],
            distribution=self.params.distribution,
            batch_std_group_size=self.batch_std_group_size,
            batch_std_num_features=self.batch_std_num_features,
            use_sn=self.use_sn_in_disc)
        self.optimizer_disc = tf.optimizers.Adam(
            self.get_learning_rate(), self.params.lr_beta1, self.params.lr_beta2)
        self.trainable_var_disc = self.discriminator.trainable_variables
        self.non_trainable_var_disc = self.discriminator.non_trainable_variables

        self.generator_synthesis = GeneratorSynthesis(
            res_out=self.image_res,
            num_channels=self.image_ch,
            num_latent=self.z_dim,
            fmap_base=8192,
            fmap_decay=1.0,
            fmap_max=self.z_dim,
            mode=self.mode,
            use_wscale=self.use_wscale,
            lr_mul=self.lr_mul['gen_synthesis'],
            distribution=self.params.distribution)
        self.generator_mapping = GeneratorMapping(
            res_out=self.image_res,
            num_input_latent=self.z_dim,
            num_mapping_layers=self.num_mapping_layers,
            num_mapping_latent=self.z_dim,
            num_output_latent=self.z_dim,
            use_wscale=self.use_wscale,
            lr_mul=self.lr_mul['gen_mapping'],
            distribution=self.params.distribution)
        self.generator_mix_style = StyleMixer(
            res_out=self.image_res,
            num_latent=self.z_dim,
            mixing_prob=self.mixing_prob,
            latent_avg_beta=self.latent_avg_beta,
            truncation_psi=self.truncation_psi,
            truncation_cutoff=self.truncation_cutoff)

        self.optimizer_gen = tf.optimizers.Adam(
            self.get_learning_rate(), self.params.lr_beta1, self.params.lr_beta2)
        self.trainable_var_gen = self.generator_synthesis.trainable_variables + \
                                 self.generator_mapping.trainable_variables + \
                                 self.generator_mix_style.trainable_variables
        self.non_trainable_var_gen = self.generator_synthesis.non_trainable_variables + \
                                     self.generator_mapping.non_trainable_variables + \
                                     self.generator_mix_style.non_trainable_variables

    @tf.function
    @convert_to_tfdata_single_batch
    @tpu_ops_decorator(mode='SUM')
    def train_disc(self, inputs, lod):
        z, images, *noises = inputs
        z2 = tf.random.normal(tf.shape(z))

        with tf.GradientTape() as tape:
            latent1 = self.generator_mapping(z)
            latent2 = self.generator_mapping(z2)
            latent = self.generator_mix_style(
                [lod, latent1, latent2], training=True)
            images_gen = self.generator_synthesis(
                [lod, latent, *noises], training=True)
            logits_fake = self.discriminator(
                [lod, images_gen], training=True)

            images_real = image_resizer(
                images, lod, res=self.image_res, mode=self.mode)
            logits_real = self.discriminator(
                [lod, images_real], training=True)

            loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(logits_fake), logits=logits_fake)
            loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(logits_real), logits=logits_real)
            base_loss = loss_fake + loss_real
            loss = tf.reduce_sum(base_loss) / self.params.batch_size

            if self.params.gp_weight != 0:
                grads, = tf.gradients(ys=logits_real, xs=images_real)
                grad_penalty = tf.reduce_sum(grads ** 2) / self.params.batch_size
                loss += 0.5 * self.params.gp_weight * grad_penalty

        grads = tape.gradient(loss, self.trainable_var_disc)
        self.optimizer_disc.apply_gradients(zip(grads, self.trainable_var_disc))
        return loss

    @tf.function
    @convert_to_tfdata_single_batch
    @tpu_ops_decorator(mode='SUM')
    def train_gen(self, inputs, lod):
        z, _, *noises = inputs
        z2 = tf.random.normal(tf.shape(z))

        with tf.GradientTape() as tape:
            latent1 = self.generator_mapping(z)
            latent2 = self.generator_mapping(z2)
            latent = self.generator_mix_style(
                [lod, latent1, latent2], training=True)
            images_gen = self.generator_synthesis(
                [lod, latent, *noises], training=True)
            logits_fake = self.discriminator(
                [lod, images_gen], training=True)

            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(logits_fake), logits=logits_fake)
            loss = tf.reduce_sum(loss) / self.params.batch_size

        grads = tape.gradient(loss, self.trainable_var_gen)
        self.optimizer_gen.apply_gradients(zip(grads, self.trainable_var_gen))
        return loss

    @tf.function
    @convert_to_tfdata_single_batch
    @tpu_ops_decorator(mode=None)
    def eval_gen(self, inputs, lod):
        z, _, *noises = inputs
        z2 = tf.random.normal(tf.shape(z))
        latent1 = self.generator_mapping(z)
        latent2 = self.generator_mapping(z2)
        latent = self.generator_mix_style(
            [lod, latent1, latent2], training=False)
        images_gen = self.generator_synthesis(
            [lod, latent, *noises], training=False)
        return images_gen

    @tpu_decorator
    def get_weights(self):
        values = self.trainable_var_gen + self.trainable_var_disc + \
                 self.non_trainable_var_gen + self.non_trainable_var_disc + \
                 self.optimizer_gen.weights + self.optimizer_disc.weights
        weights = {}
        for i, v in enumerate(values):
            key = v.name
            weights[key] = v
        return weights

    @tpu_decorator
    def set_weights(self, weights, load_optimizer=True):
        tensors = self.trainable_var_gen + self.trainable_var_disc + \
                  self.non_trainable_var_gen + self.non_trainable_var_disc + \
                  self.optimizer_gen.weights + self.optimizer_disc.weights
        for tensor in tensors:
            if not load_optimizer and 'Adam' in tensor.name:
                print('Skip ' + tensor.name + ' ...')
                continue
            elif tensor.name in weights.keys():
                tensor.assign(weights[tensor.name])
            else:
                print('Skip ' + tensor.name + ' ...')

    @tf.function
    @tpu_decorator
    def initialize_optimizer(self):
        vars = self.discriminator.weights + self.optimizer_gen.weights
        for var in vars:
            if 'iter' in var.name:
                continue
            elif 'Adam' in var.name:
                var.assign(tf.zeros_like(var))
