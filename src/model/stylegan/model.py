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

        print('build Discriminator...')
        self.discriminator = Discriminator(
            res=self.image_res,
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
            self.get_learning_rate(), self.params.lr_beta1, self.params.lr_beta2,
            name='Adam_disc')

        print('build Generator Synthesis...')
        self.generator_synthesis = GeneratorSynthesis(
            res_out=self.image_res,
            num_latent=self.z_dim,
            fmap_base=8192,
            fmap_decay=1.0,
            fmap_max=self.z_dim,
            mode=self.mode,
            use_wscale=self.use_wscale,
            lr_mul=self.lr_mul['gen_synthesis'],
            distribution=self.params.distribution)
        print('build Generator Mapping...')
        self.generator_mapping = GeneratorMapping(
            res_out=self.image_res,
            num_mapping_layers=self.num_mapping_layers,
            num_mapping_latent=self.z_dim,
            num_input_latent=self.z_dim,
            num_output_latent=self.z_dim,
            use_wscale=self.use_wscale,
            lr_mul=self.lr_mul['gen_mapping'],
            distribution=self.params.distribution)
        print('build Generator Style Mixer...')
        self.generator_mix_style = StyleMixer(
            res_out=self.image_res,
            num_latent=self.z_dim,
            mixing_prob=self.mixing_prob,
            latent_avg_beta=self.latent_avg_beta,
            truncation_psi=self.truncation_psi,
            truncation_cutoff=self.truncation_cutoff)

        self.optimizer_gen = tf.optimizers.Adam(
            self.get_learning_rate(), self.params.lr_beta1, self.params.lr_beta2,
            name='Adam_gen')

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

            with tf.GradientTape() as tape2:
                tape2.watch(images)
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

            if self.params.gp_weight != 0.:
                grads = tape2.gradient(logits_real, images)
                grad_penalty = tf.reduce_sum(grads ** 2) / self.params.batch_size
                loss += 0.5 * self.params.gp_weight * grad_penalty

        trainable_vars = self.discriminator.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer_disc.apply_gradients(zip(grads, trainable_vars))
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

        trainable_vars = self.generator_synthesis.trainable_variables + \
                         self.generator_mapping.trainable_variables + \
                         self.generator_mix_style.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer_gen.apply_gradients(zip(grads, trainable_vars))
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
        model_weights = {}
        gen_models = [self.generator_mapping,
                      self.generator_mix_style,
                      self.generator_synthesis]
        disc_models = [self.discriminator]
        for model in gen_models + disc_models:
            model_weights[model.name] = {}
            for v in model.weights:
                model_weights[model.name][v.name] = v.numpy()

        opt_weights = {}
        optimizers = [self.optimizer_gen, self.optimizer_disc]
        for opt, models in zip(optimizers, [gen_models, disc_models]):
            slot_names = opt.get_slot_names()
            opt_weights[opt._name] = {}
            for model in models:
                opt_weights[opt._name][model.name] = {}
                for v in model.trainable_variables:
                    weights_per_var = {}
                    for slot_name in slot_names:
                        weights_per_var[slot_name] = opt.get_slot(v, slot_name).numpy()
                    opt_weights[opt._name][model.name][v.name] = weights_per_var

        return {'model': model_weights, 'optimizer': opt_weights}

    def _set_model_weights(self, model, weights):
        for tensor in model.weights:
            if tensor.name in weights.keys():
                tensor.assign(weights[tensor.name])
            else:
                print('Skip ' + tensor.name + ' ...')

        tensors_list = [t.name for t in model.weights]
        for key in weights.keys():
            if key not in tensors_list:
                print('Not loaded ' + key + ' ...')

    def _set_optimizer_weights(self, model, opt, weights):
        for v in model.trainable_variables:
            if v.name not in weights.keys():
                print('Skip optimizer weights of ' + v.name + ' ...')
                continue
            v_opt = weights[v.name]
            for slot_name, v_opt_slot in v_opt.items():
                initializer = tf.initializers.Constant(v_opt_slot)
                opt.add_slot(v, slot_name, initializer=initializer)

    @tpu_decorator
    def set_weights(self, weights, load_optimizer=True):
        optimizers = [self.optimizer_gen, self.optimizer_disc]
        gen_models = [self.generator_mapping,
                      self.generator_mix_style,
                      self.generator_synthesis]
        disc_models = [self.discriminator]
        opt_weights = weights['optimizer']
        model_weights = weights['model']

        for opt, models in zip(optimizers, [gen_models, disc_models]):
            opt_name = opt._name
            with tf.name_scope(opt_name):
                for model in models:
                    self._set_model_weights(model, model_weights[model.name])
                    if load_optimizer:
                        self._set_optimizer_weights(
                            model, opt, opt_weights[opt_name][model.name])

    @tf.function
    @tpu_decorator
    def initialize_optimizer(self):
        vars = self.optimizer_disc.weights + self.optimizer_gen.weights
        for var in vars:
            if 'iter' in var.name:
                continue
            var.assign(tf.zeros_like(var))
