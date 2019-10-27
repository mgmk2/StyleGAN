import numpy as np
import tensorflow as tf

from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers \
    import Conv2D, Conv2DTranspose, Dense, BatchNormalization,\
           LeakyReLU, Reshape, Flatten, Lambda, RepeatVector, AveragePooling2D

from ...layers \
    import ScaleAddToConst, ScaleAdd, AdaIN, AddBias2D, Blur, MixStyle, \
           UpSampling2D, PixelNormalization, BatchStddev, SNDense, SNConv2D, \
           ScaledDense, ScaledConv2D
from ...utils.utils import num_div2

def res2num_blocks(res):
    return num_div2(res) - 1

def interpolate_clip(x1, x2, ratio):
    ratio = tf.clip_by_value(ratio, 0.0, 1.0)
    rank = tf.rank(x1)
    shape = tf.concat([[-1], tf.ones(rank - 1, tf.int64)], axis=0)
    ratio = tf.reshape(ratio, shape)
    return x1 + ratio * (x2 - x1)

def get_initializer(distribution, use_wscale=True, relu_alpha=0):
    if use_wscale:
        if distribution in ['normal', 'truncated_normal']:
            std = 1 / 0.87962566103423978
            return tf.initializers.TruncatedNormal(0, std)
        if distribution == 'untruncated_normal':
            return tf.initializers.RandomNormal(0, 1)
        else:
            return tf.initializers.RandomUniform(-np.sqrt(3), np.sqrt(3))

    scale = 2 / (1 + relu_alpha ** 2)
    return tf.initializers.VarianceScaling(
        scale, mode='fan_in', distribution=distribution)

#===============================================================================

def image_resizer(image, lod, res=32, mode=None):
    if mode is None: mode = 'dynamic'
    num_blocks = res2num_blocks(res)
    with tf.name_scope('image_resizer'):
        lod = tf.cast(lod, tf.float32)
        lod = tf.reshape(lod, [-1])[0]

        if mode == 'static':
            x = [None for _ in range(num_blocks)]
            x[0] = image
            for i in range(1, num_blocks):
                res = res // 2
                ksize = (1, 2, 2, 1)
                x[i] = tf.nn.avg_pool(x[i - 1], ksize, strides=ksize, padding='VALID')

            y = x[-1]
            for i in range(1, num_blocks):
                lod_i = tf.constant(i, tf.float32)
                y = tf.reshape(y, [-1, res, 1, res, 1, 3])
                y = tf.tile(y, [1, 1, 2, 1, 2, 1])
                y = tf.reshape(y, [-1, 2 * res, 2 * res, 3])
                y = interpolate_clip(x[-(i + 1)], y, lod_i - lod)
                res *= 2

        elif mode == 'dynamic':
            lod_int = tf.cast(tf.math.ceil(lod), tf.int64)
            s = tf.shape(image, out_type=tf.int64)
            factor = 2 ** (num_blocks - lod_int - 1)
            x = tf.reshape(image, [-1, s[1] // factor, factor, s[2] // factor, factor, s[3]])
            x = tf.reduce_mean(x, axis=[2, 4])

            sx = tf.shape(x, out_type=tf.int64)
            h = tf.reshape(x, [-1, sx[1] // 2, 2, sx[2] // 2, 2, sx[3]])
            h = tf.reduce_mean(h, axis=[2, 4], keepdims=True)
            h = tf.tile(h, [1, 1, 2, 1, 2, 1])
            x2 = tf.reshape(h, [-1, sx[1], sx[2], sx[3]])
            y = interpolate_clip(x, x2, tf.cast(lod_int, tf.float32) - lod)

            y = tf.reshape(y, [-1, sx[1], 1, sx[2], 1, sx[3]])
            y = tf.tile(y, [1, 1, factor, 1, factor, 1])
            y = tf.reshape(y, [-1, s[1], s[2], s[3]])
        return y

#===============================================================================

class AdaIN_block(Model):
    def __init__(self,
                 res,
                 num_channels,
                 num_latent,
                 use_wscale=True,
                 lr_mul=1.0,
                 distribution='untruncated_normal',
                 **kwargs):
        super(AdaIN_block, self).__init__(**kwargs)
        self.res = res
        self.num_channels = num_channels
        self.num_latent = num_latent

        with tf.name_scope(self.name) as scope:
            self.dense = ScaledDense(
                2 * num_channels,
                use_wscale=use_wscale,
                lr_mul=lr_mul,
                kernel_initializer=get_initializer(
                    distribution=distribution, use_wscale=use_wscale),
                name=scope + 'scaled_dense_{0:}x{0:}'.format(res))
            self.reshape_layer = Reshape((2, 1, 1, num_channels))
            self.adain_layer = AdaIN()
            self.initialize_layers()

    def initialize_layers(self):
        x = Input((self.res, self.res, self.num_channels))
        w = Input((self.num_latent,))
        _ = self.call((x, w))

    def call(self, inputs):
        x, w = inputs
        style = self.dense(w)
        style = self.reshape_layer(style)
        y = self.adain_layer((x, style))
        return y

class const_block(Model):
    def __init__(self,
                 res,
                 num_filters,
                 num_latent,
                 use_wscale=True,
                 lr_mul=1.0,
                 distribution='untruncated_normal',
                 **kwargs):
        super(const_block, self).__init__(**kwargs)
        self.res = res
        self.num_latent = num_latent

        with tf.name_scope(self.name) as scope:
            self.slice_noise0 = Lambda(lambda x: x[:, :, :, 0])
            self.slice_noise1 = Lambda(lambda x: x[:, :, :, 1])
            self.slice_w0 = Lambda(lambda x: x[:, 0])
            self.slice_w1 = Lambda(lambda x: x[:, 1])
            self.scaleadd_to_const = ScaleAddToConst((res, res, num_filters))
            self.add_bias0 = AddBias2D(
                name=scope + 'add_bias2d_{0:}x{0:}_0'.format(res))
            self.act0 = LeakyReLU(alpha=0.2)
            self.adain0 = AdaIN_block(
                res, num_filters, num_latent, use_wscale=use_wscale, lr_mul=lr_mul,
                distribution=distribution, name='AdaIN_block_{0:}x{0:}_0'.format(res))

            self.scaled_conv = ScaledConv2D(
                num_filters,
                (3, 3),
                padding='same',
                use_wscale=use_wscale,
                lr_mul=lr_mul,
                kernel_initializer=get_initializer(
                    distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
                name=scope + 'scaled_conv2d_{0:}x{0:}'.format(res))
            self.scale_add = ScaleAdd(
                name=scope + 'scale_add_{0:}x{0:}'.format(res))
            self.add_bias1 = AddBias2D(
                name=scope + 'add_bias2d_{0:}x{0:}_1'.format(res))
            self.act1 = LeakyReLU(alpha=0.2)
            self.adain1 = AdaIN_block(
                res, num_filters, num_latent, use_wscale=use_wscale, lr_mul=lr_mul,
                distribution=distribution, name='AdaIN_block_{0:}x{0:}_1'.format(res))
            self.initialize_layers()

    def initialize_layers(self):
        w = Input((2, self.num_latent))
        noise = Input((self.res, self.res, 2))
        _ = self.call((w, noise))

    def call(self, inputs):
        w, noise = inputs
        noise0 = self.slice_noise0(noise)
        noise1 = self.slice_noise1(noise)
        w0 = self.slice_w0(w)
        w1 = self.slice_w1(w)

        h = self.scaleadd_to_const(noise0)
        h = self.add_bias0(h)
        h = self.act0(h)
        h = self.adain0((h, w0))

        h = self.scaled_conv(h)
        h = self.scale_add((h, noise1))
        h = self.add_bias1(h)
        h = self.act1(h)
        y = self.adain1((h, w1))
        return y

class generator_block(Model):
    def __init__(self,
                 input_shape,
                 res,
                 num_filters,
                 num_latent,
                 use_wscale=True,
                 lr_mul=1.0,
                 distribution='untruncated_normal',
                 **kwargs):
        super(generator_block, self).__init__(**kwargs)
        self.x_shape = input_shape
        self.res = res
        self.num_latent = num_latent

        with tf.name_scope(self.name) as scope:
            self.slice_noise0 = Lambda(lambda x: x[:, :, :, 0])
            self.slice_noise1 = Lambda(lambda x: x[:, :, :, 1])
            self.slice_w0 = Lambda(lambda x: x[:, 0])
            self.slice_w1 = Lambda(lambda x: x[:, 1])

            self.upsampling = UpSampling2D(
                (2, 2), name='upsampling2d_{0:}x{0:}'.format(res))
            self.scaled_conv0 = ScaledConv2D(
                num_filters,
                (3, 3),
                padding='same',
                use_bias=False,
                use_wscale=use_wscale,
                lr_mul=lr_mul,
                kernel_initializer=get_initializer(
                    distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
                name=scope + 'scaled_conv2d_{0:}x{0:}_0'.format(res))
            self.blur = Blur()
            self.scale_add0 = ScaleAdd(
                name=scope + 'scale_add_{0:}x{0:}_0'.format(res))
            self.add_bias0 = AddBias2D(
                name=scope + 'add_bias2d_{0:}x{0:}_0'.format(res))
            self.act0 = LeakyReLU(alpha=0.2)
            self.adain0 = AdaIN_block(
                res, num_filters, num_latent, use_wscale=use_wscale, lr_mul=lr_mul,
                distribution=distribution, name='AdaIN_block_{0:}x{0:}_0'.format(res))

            self.scaled_conv1 = ScaledConv2D(
                num_filters,
                (3, 3),
                padding='same',
                use_bias=False,
                use_wscale=use_wscale,
                lr_mul=lr_mul,
                kernel_initializer=get_initializer(
                    distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
                name=scope + 'scaled_conv2d_{0:}x{0:}_1'.format(res))
            self.scale_add1 = ScaleAdd(
                name=scope + 'scale_add_{0:}x{0:}_1'.format(res))
            self.add_bias1 = AddBias2D(
                name=scope + 'add_bias2d_{0:}x{0:}_1'.format(res))
            self.act1 = LeakyReLU(alpha=0.2)
            self.adain1 = AdaIN_block(
                res, num_filters, num_latent, use_wscale=use_wscale, lr_mul=lr_mul,
                distribution=distribution, name='AdaIN_block_{0:}x{0:}_1'.format(res))
            self.initialize_layers()

    def initialize_layers(self):
        x = Input(self.x_shape)
        w = Input((2, self.num_latent))
        noise = Input((self.res, self.res, 2))
        _ = self.call((x, w, noise))

    def call(self, inputs):
        x, w, noise = inputs
        noise0 = self.slice_noise0(noise)
        noise1 = self.slice_noise1(noise)
        w0 = self.slice_w0(w)
        w1 = self.slice_w1(w)

        h = self.upsampling(x)
        h = self.scaled_conv0(h)
        h = self.blur(h)
        h = self.scale_add0((h, noise0))
        h = self.add_bias0(h)
        h = self.act0(h)
        h = self.adain0((h, w0))

        h = self.scaled_conv1(h)
        h = self.scale_add1((h, noise1))
        h = self.add_bias1(h)
        h = self.act1(h)
        y = self.adain1((h, w1))
        return y

class toRGB(Model):
    def __init__(self,
                 input_shape,
                 res,
                 num_channels,
                 use_wscale=True,
                 lr_mul=1.0,
                 distribution='untruncated_normal',
                 **kwargs):
        super(toRGB, self).__init__(**kwargs)
        self.x_shape = input_shape
        self.res = res

        with tf.name_scope(self.name) as scope:
            self.scaled_conv = ScaledConv2D(
                num_channels,
                (1, 1),
                padding='same',
                lr_mul=lr_mul,
                kernel_initializer=get_initializer(
                    distribution=distribution, use_wscale=use_wscale),
                name=scope + 'scaled_conv2d_{0:}x{0:}_toRGB'.format(res))
            self.initialize_layers()

    def initialize_layers(self):
        inputs = Input(self.x_shape)
        _ = self.call(inputs)

    def call(self, inputs):
        with tf.name_scope('toRGB_{0:}x{0:}'.format(self.res)):
            y = self.scaled_conv(inputs)
        return y

class GeneratorSynthesis(Model):
    def __init__(self, res_out=32,
                num_channels=3,
                num_latent=512,
                fmap_base=8192,
                fmap_decay=1.0,
                fmap_max=512,
                mode=None,
                use_wscale=True,
                lr_mul=1.0,
                distribution='untruncated_normal',
                **kwargs):
        super(GeneratorSynthesis, self).__init__(**kwargs)
        self.num_channels = num_channels
        self.num_latent = num_latent
        self.fmap_base = fmap_base
        self.fmap_decay = fmap_decay
        self.fmap_max = fmap_max

        if mode is not None and mode not in ['dynamic', 'static']:
            raise ValueError('Unknown mode: ' + mode)
        self.mode = 'dynamic' if mode is None else mode

        self.use_wscale = use_wscale
        self.lr_mul = lr_mul
        self.distribution = distribution

        self.num_blocks = res2num_blocks(res_out)
        self.mylayers = {}
        self._build_layers()

    def res2num_filters(self, res):
        return min(
            int(self.fmap_base * (2.0 / res) ** self.fmap_decay),
            self.fmap_max)

    def _build_layers(self):
        with tf.name_scope('generator_synthesis') as scope:
            res = 4
            input_shape = (res, res, self.res2num_filters(res))
            self.const_block = const_block(
                res, self.res2num_filters(res), self.num_latent,
                use_wscale=self.use_wscale, lr_mul=self.lr_mul,
                distribution=self.distribution,
                name='const_block')
            self.image_out_layer0 = toRGB(
                input_shape, res, self.num_channels, use_wscale=self.use_wscale,
                lr_mul=self.lr_mul, distribution=self.distribution,
                name='toRGB_{0:}x{0:}'.format(res))

            for i in range(1, self.num_blocks):
                input_shape = (res, res, self.res2num_filters(res))
                res *= 2
                setattr(self, 'block{:}'.format(i), generator_block(
                    input_shape, res, self.res2num_filters(res),
                    self.num_latent, use_wscale=self.use_wscale,
                    lr_mul=self.lr_mul, distribution=self.distribution,
                    name='generator_block_{0:}x{0:}'.format(res)))

                input_shape = (res, res, self.res2num_filters(res))
                setattr(self, 'toRGB{:}'.format(i), toRGB(
                    input_shape, res, self.num_channels, use_wscale=self.use_wscale,
                    lr_mul=self.lr_mul, distribution=self.distribution,
                    name='toRGB_{0:}x{0:}'.format(res)))

                setattr(self, 'image_out_layer{:}'.format(i), UpSampling2D(
                    (2, 2), name='upsampling2d_image_out_{0:}x{0:}'.format(res)))

                setattr(self, 'x_out_layer{:}'.format(i), UpSampling2D(
                    (2, 2), name='upsampling2d_x_out_{0:}x{0:}'.format(res)))

    def call(self, inputs, training=None):
        lod, w, *noise = inputs
        lod = tf.reshape(lod, [-1])[0]

        w0 = w[:, :2]
        x = self.const_block((w0, noise[0]))
        image_out = self.image_out_layer0(x)

        if self.mode == 'static':
            for i in range(1, self.num_blocks):
                lod_i = tf.cast(i, tf.float32)
                image_out = getattr(self, 'image_out_layer{:}'.format(i))(image_out)
                w_i = w[:, i * 2:(i + 1) * 2]
                x = getattr(self, 'block{:}'.format(i))((x, w_i, noise[i]))
                y = getattr(self, 'toRGB{:}'.format(i))(x)
                image_out = interpolate_clip(y, image_out, lod_i - lod)

        elif self.mode == 'dynamic':
            for i in range(1, self.num_blocks):
                lod_i = tf.cast(i, tf.float32)
                @tf.function
                def block_i(x, image_out, w, noise, lod):
                    if lod_i >= lod + 1:
                        x = getattr(self, 'x_out_layer{:}'.format(i))(x)
                        image_out = getattr(
                            self, 'image_out_layer{:}'.format(i))(image_out)
                    elif lod_i <= lod:
                        w_i = w[:, i * 2:(i + 1) * 2]
                        x = getattr(self, 'block{:}'.format(i))((x, w_i, noise[i]))
                        image_out = getattr(self, 'toRGB{:}'.format(i))(x)
                    else:
                        w_i = w[:, i * 2:(i + 1) * 2]
                        x = getattr(self, 'block{:}'.format(i))((x, w_i, noise[i]))
                        x_new = getattr(self, 'toRGB{:}'.format(i))(x)
                        image_out_new = getattr(
                            self, 'image_out_layer{:}'.format(i))(image_out)
                        image_out = interpolate_clip(
                            x_new, image_out_new, lod_i - lod)
                    return x, image_out
                x, image_out = block_i(x, image_out, w, noise, lod)
        return image_out

class GeneratorMapping(Model):
    def __init__(self,
                 res_out=32,
                 num_mapping_layers=8,
                 num_mapping_latent=512,
                 num_input_latent=512,
                 num_output_latent=512,
                 use_wscale=True,
                 lr_mul=1.0,
                 distribution='untruncated_normal',
                 **kwargs):
        super(GeneratorMapping, self).__init__(**kwargs)
        self.num_mapping_layers = num_mapping_layers
        self.num_input_latent = num_input_latent

        num_blocks = res2num_blocks(res_out)
        num_repeat_output = 2 * num_blocks

        with tf.name_scope('generator_mapping') as scope:
            self.pixel_norm = PixelNormalization()
            self.repeat_vector = RepeatVector(num_repeat_output)

            for i in range(num_mapping_layers):
                if i == num_mapping_layers - 1:
                    num_latent = num_output_latent
                else:
                    num_latent = num_mapping_latent

                setattr(self, 'scaled_dense{:}'.format(i), ScaledDense(
                    num_latent,
                    use_wscale=use_wscale,
                    lr_mul=lr_mul,
                    kernel_initializer=get_initializer(
                        distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
                    name=scope + 'scaled_dense_{:}'.format(i)))
                setattr(self, 'act{:}'.format(i), LeakyReLU(alpha=0.2))
            self.initialize_layers()

    def initialize_layers(self):
        inputs = Input((self.num_input_latent,))
        _ = self.call(inputs)

    def call(self, inputs):
        h = self.pixel_norm(inputs)
        for i in range(self.num_mapping_layers):
            h = getattr(self, 'scaled_dense{:}'.format(i))(h)
            h = getattr(self, 'act{:}'.format(i))(h)
        outputs = self.repeat_vector(h)
        return outputs

class StyleMixer(Model):
    def __init__(self, res_out=32,
                 num_latent=512,
                 mixing_prob=0.9,
                 latent_avg_beta=0.995,
                 truncation_psi=0.7,
                 truncation_cutoff=8,
                 **kwargs):
        super(StyleMixer, self).__init__(**kwargs)
        self.num_latent = num_latent
        num_blocks = res2num_blocks(res_out)
        self.num_layers = 2 * num_blocks

        with tf.name_scope('generator_mix_style') as scope:
            self.reshape_layer = Reshape((1, 1, 1))
            self.mix_style = MixStyle(
                self.num_layers,
                mixing_prob=mixing_prob,
                latent_avg_beta=latent_avg_beta,
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff,
                name=scope + 'mix_style')
            self.initialize_layers()

    def initialize_layers(self):
        latent1 = Input((self.num_layers, self.num_latent,))
        latent2 = Input((self.num_layers, self.num_latent,))
        lod = Input((1,))
        _ = self.call((lod, latent1, latent2))

    def call(self, inputs):
        lod, latent1, latent2 = inputs
        lod_tensor = self.reshape_layer(lod)
        return self.mix_style((latent1, latent2, lod_tensor))

#===============================================================================

class discriminator_block_output(Model):
    def __init__(self,
                 input_shape,
                 res,
                 num_filters,
                 use_wscale=True,
                 lr_mul=1.0,
                 distribution='untruncated_normal',
                 batch_std_group_size=4,
                 batch_std_num_features=1,
                 use_sn=False,
                 **kwargs):
        super(discriminator_block_output, self).__init__(**kwargs)
        self.x_shape = input_shape

        with tf.name_scope(self.name) as scope:
            self.batch_stddev = BatchStddev(
                group_size=batch_std_group_size, num_features=batch_std_num_features)
            self.act0 = LeakyReLU(alpha=0.2)
            self.act1 = LeakyReLU(alpha=0.2)
            self.flatten = Flatten()

            if use_sn:
                self.conv = SNConv2D(
                    num_filters[0],
                    (3, 3),
                    padding='same',
                    use_wscale=use_wscale,
                    lr_mul=lr_mul,
                    kernel_initializer=get_initializer(
                        distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
                    name=scope + 'sn_conv2d_{0:}x{0:}'.format(res))
                self.dense0 = SNDense(
                    num_filters[1],
                    use_wscale=use_wscale,
                    lr_mul=lr_mul,
                    kernel_initializer=get_initializer(
                        distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
                    name=scope + 'sn_dense_{0:}x{0:}_0'.format(res))
                self.dense1 = SNDense(
                    1,
                    use_wscale=use_wscale,
                    lr_mul=lr_mul,
                    kernel_initializer=get_initializer(
                        distribution=distribution, use_wscale=use_wscale),
                    name=scope + 'sn_dense_{0:}x{0:}_1'.format(res))

            else:
                self.conv = ScaledConv2D(
                    num_filters[0],
                    (3, 3),
                    padding='same',
                    use_wscale=use_wscale,
                    lr_mul=lr_mul,
                    kernel_initializer=get_initializer(
                        distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
                    name=scope + 'scaled_conv2d_{0:}x{0:}'.format(res))
                self.dense0 = ScaledDense(
                    num_filters[1],
                    use_wscale=use_wscale,
                    lr_mul=lr_mul,
                    kernel_initializer=get_initializer(
                        distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
                    name=scope + 'scaled_dense_{0:}x{0:}_0'.format(res))
                self.dense1 = ScaledDense(
                    1,
                    use_wscale=use_wscale,
                    lr_mul=lr_mul,
                    kernel_initializer=get_initializer(
                        distribution=distribution, use_wscale=use_wscale),
                    name=scope + 'scaled_dense_{0:}x{0:}_1'.format(res))
            self.initialize_layers()

    def initialize_layers(self):
        inputs = Input(self.x_shape)
        _ = self.call(inputs)

    def call(self, inputs):
        h = self.batch_stddev(inputs)
        h = self.conv(h)
        h = self.act0(h)
        h = self.flatten(h)
        h = self.dense0(h)
        h = self.act1(h)
        y = self.dense1(h)
        return y

class discriminator_block(Model):
    def __init__(self,
                 input_shape,
                 res,
                 num_filters,
                 use_wscale=True,
                 lr_mul=1.0,
                 distribution='untruncated_normal',
                 use_sn=False,
                 **kwargs):
        super(discriminator_block, self).__init__(**kwargs)
        self.x_shape = input_shape

        with tf.name_scope(self.name) as scope:
            self.act0 = LeakyReLU(alpha=0.2)
            self.act1 = LeakyReLU(alpha=0.2)
            self.blur = Blur()
            self.down_sample = AveragePooling2D((2, 2))
            self.add_bias = AddBias2D(
                name=scope + 'add_bias2d_{0:}x{0:}'.format(res))

            if use_sn:
                self.conv0 = SNConv2D(
                    num_filters[0],
                    (3, 3),
                    padding='same',
                    use_wscale=use_wscale,
                    lr_mul=lr_mul,
                    kernel_initializer=get_initializer(
                        distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
                    name=scope + 'sn_conv2d_{0:}x{0:}_0'.format(res))
                self.conv1 = SNConv2D(
                    num_filters[1],
                    (3, 3),
                    padding='same',
                    use_wscale=use_wscale,
                    lr_mul=lr_mul,
                    kernel_initializer=get_initializer(
                        distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
                    name=scope + 'sn_conv2d_{0:}x{0:}_1'.format(res))

            else:
                self.conv0 = ScaledConv2D(
                    num_filters[0],
                    (3, 3),
                    padding='same',
                    use_wscale=use_wscale,
                    lr_mul=lr_mul,
                    kernel_initializer=get_initializer(
                        distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
                    name=scope + 'scaled_conv2d_{0:}x{0:}_0'.format(res))
                self.conv1 = ScaledConv2D(
                    num_filters[1],
                    (3, 3),
                    padding='same',
                    use_wscale=use_wscale,
                    lr_mul=lr_mul,
                    kernel_initializer=get_initializer(
                        distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
                    name=scope + 'scaled_conv2d_{0:}x{0:}_1'.format(res))
            self.initialize_layers()

    def initialize_layers(self):
        inputs = Input(self.x_shape)
        _ = self.call(inputs)

    def call(self, inputs):
        h = self.conv0(inputs)
        h = self.act0(h)
        h = self.blur(h)

        h = self.conv1(h)
        h = self.down_sample(h)
        h = self.add_bias(h)
        y = self.act1(h)
        return y

class fromRGB(Model):
    def __init__(self,
                 input_shape,
                 res,
                 num_filters,
                 use_wscale=True,
                 lr_mul=1.0,
                 distribution='untruncated_normal',
                 use_sn=False,
                 **kwargs):
        super(fromRGB, self).__init__(**kwargs)
        self.x_shape = input_shape

        with tf.name_scope(self.name) as scope:
            self.act = LeakyReLU(alpha=0.2)
            if use_sn:
                self.conv = SNConv2D(
                    num_filters,
                    (1, 1),
                    padding='same',
                    use_wscale=use_wscale,
                    lr_mul=lr_mul,
                    kernel_initializer=get_initializer(
                        distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
                    name=scope + 'sn_conv2d_{0:}x{0:}_fromRGB'.format(res))
            else:
                self.conv = ScaledConv2D(
                    num_filters,
                    (1, 1),
                    padding='same',
                    use_wscale=use_wscale,
                    lr_mul=lr_mul,
                    kernel_initializer=get_initializer(
                        distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
                    name=scope + 'scale_conv2d_{0:}x{0:}_fromRGB'.format(res))
            self.initialize_layers()

    def initialize_layers(self):
        inputs = Input(self.x_shape)
        _ = self.call(inputs)

    def call(self, inputs):
        h = self.conv(inputs)
        y = self.act(h)
        return y

class Discriminator(Model):
    def __init__(self,
                res=32,
                num_channels=3,
                fmap_base=8192,
                fmap_decay=1.0,
                fmap_max=512,
                mode=None,
                use_wscale=True,
                lr_mul=1.0,
                distribution='untruncated_normal',
                batch_std_group_size=4,
                batch_std_num_features=1,
                use_sn=False,
                **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.res = res
        self.num_channels = num_channels
        self.fmap_base = fmap_base
        self.fmap_decay = fmap_decay
        self.fmap_max = fmap_max

        if mode is not None and mode not in ['dynamic', 'static']:
            raise ValueError('Unknown mode: ' + mode)
        self.mode = 'dynamic' if mode is None else mode

        self.use_wscale = use_wscale
        self.lr_mul = lr_mul
        self.distribution = distribution

        self.batch_std_group_size = batch_std_group_size
        self.batch_std_num_features = batch_std_num_features

        self.use_sn = use_sn

        self.num_blocks = res2num_blocks(res)
        self._build_layers()

    def res2num_filters(self, res):
        return min(int(self.fmap_base * (2.0 / res) ** self.fmap_decay), self.fmap_max)

    def _build_layers(self):
        res = self.res
        input_shape = (res, res, self.num_channels)

        with tf.name_scope('discriminator'):
            self.fromRGB0 = fromRGB(
                input_shape, res, self.res2num_filters(res), use_wscale=self.use_wscale,
                lr_mul=self.lr_mul, distribution=self.distribution, use_sn=self.use_sn,
                name='fromRGB_{0:}x{0:}'.format(res))

            for k in range(1, self.num_blocks):
                i = self.num_blocks - k
                input_shape = (res, res, self.res2num_filters(res))
                num_filters = (self.res2num_filters(res), self.res2num_filters(res // 2))
                setattr(self, 'down_sample{:}'.format(k), AveragePooling2D(
                    (2, 2), name='down_sample_{0:}x{0:}'.format(res)))
                setattr(self, 'block{:}'.format(k), discriminator_block(
                    input_shape, res, num_filters, use_wscale=self.use_wscale,
                    lr_mul=self.lr_mul, distribution=self.distribution, use_sn=self.use_sn,
                    name='discriminator_block_{0:}x{0:}'.format(res)))

                res = res // 2
                input_shape = (res, res, self.num_channels)
                setattr(self, 'fromRGB{:}'.format(k), fromRGB(
                    input_shape, res, self.res2num_filters(res),
                    use_wscale=self.use_wscale, lr_mul=self.lr_mul,
                    distribution=self.distribution, use_sn=self.use_sn,
                    name='fromRGB_{0:}x{0:}'.format(res)))

            input_shape = (res, res, self.res2num_filters(res))
            num_filters = (self.res2num_filters(res), self.res2num_filters(res // 2))
            self.output_layer = discriminator_block_output(
                input_shape, res, num_filters, use_wscale=self.use_wscale,
                lr_mul=self.lr_mul, distribution=self.distribution,
                batch_std_group_size=self.batch_std_group_size,
                batch_std_num_features=self.batch_std_num_features,
                use_sn=self.use_sn, name='discriminator_block_output')

    def call(self, inputs, training=None):
        lod, image = inputs
        lod = tf.reshape(lod, [-1])[0]

        x = self.fromRGB0(image)

        if self.mode == 'static':
            for k in range(1, self.num_blocks):
                lod_k = tf.cast(self.num_blocks - k, tf.float32)
                image = getattr(self, 'down_sample{:}'.format(k))(image)
                y = getattr(self, 'fromRGB{:}'.format(k))(image)
                x = getattr(self, 'block{:}'.format(k))(x)
                x = interpolate_clip(x, y, lod_k - lod)

        elif self.mode == 'dynamic':
            for k in range(1, self.num_blocks):
                lod_k = tf.cast(self.num_blocks - k, tf.float32)
                image = getattr(self, 'down_sample{:}'.format(k))(image)
                @tf.function
                def block_i(x, image, lod):
                    if lod_k >= lod + 1:
                        x = getattr(self, 'fromRGB{:}'.format(k))(image)
                    elif lod_k <= lod:
                        x = getattr(self, 'block{:}'.format(k))(x)
                    else:
                        x_new = getattr(self, 'block{:}'.format(k))(x)
                        y_new = getattr(self, 'fromRGB{:}'.format(k))(image)
                        x = interpolate_clip(x_new, y_new, lod_k - lod)
                    return x
                x = block_i(x, image, lod)
        outputs = self.output_layer(x)
        return outputs
