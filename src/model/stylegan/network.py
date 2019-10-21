import numpy as np
import tensorflow as tf

from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers \
    import Conv2D, Conv2DTranspose, Dense, BatchNormalization,\
           LeakyReLU, Reshape, Flatten, Lambda, RepeatVector, Layer

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

def AdaIN_block(res, num_channels, num_latent, use_wscale=True, lr_mul=1.0, distribution='untruncated_normal'):
    with tf.name_scope('AdaIN_block_{0:}x{0:}'.format(res)):
        x = Input((res, res, num_channels))
        w = Input((num_latent,))

        style = ScaledDense(
            2 * num_channels,
            use_wscale=use_wscale,
            lr_mul=lr_mul,
            kernel_initializer=get_initializer(
                distribution=distribution, use_wscale=use_wscale),
            name='scaled_dense_{0:}x{0:}'.format(res))(w)
        style = Reshape([2, 1, 1, num_channels])(style)
        y = AdaIN()([x, style])
        return Model([x, w], y)

def const_block(res, num_filters, num_latent, use_wscale=True, lr_mul=1.0, distribution='untruncated_normal'):

    with tf.name_scope('const_block'):
        w = Input((2, num_latent))
        noise = Input((res, res, 2))

        noise0 = Lambda(lambda x: x[:, :, :, 0])(noise)
        noise1 = Lambda(lambda x: x[:, :, :, 1])(noise)
        w0 = Lambda(lambda x: x[:, 0])(w)
        w1 = Lambda(lambda x: x[:, 1])(w)

        h = ScaleAddToConst((res, res, num_filters))(noise0)
        h = AddBias2D(name='add_bias2d_{0:}x{0:}_0'.format(res))(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = AdaIN_block(
            res, num_filters, num_latent,
            use_wscale=use_wscale, lr_mul=lr_mul, distribution=distribution)([h, w0])

        h = ScaledConv2D(
            num_filters,
            (3, 3),
            padding='same',
            use_wscale=use_wscale,
            lr_mul=lr_mul,
            kernel_initializer=get_initializer(
                distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
            name='scaled_conv2d_{0:}x{0:}'.format(res))(h)
        h = ScaleAdd(name='scale_add_{0:}x{0:}'.format(res))([h, noise1])
        h = AddBias2D(name='add_bias2d_{0:}x{0:}_1'.format(res))(h)
        h = LeakyReLU(alpha=0.2)(h)
        y = AdaIN_block(
            res, num_filters, num_latent,
            use_wscale=use_wscale, lr_mul=lr_mul, distribution=distribution)([h, w1])
        return Model([w, noise], y)

def generator_block(x_shape, res, num_filters, num_latent, use_wscale=True, lr_mul=1.0, distribution='untruncated_normal'):

    with tf.name_scope('generator_block_{0:}x{0:}'.format(res)):
        x = Input(x_shape)
        w = Input((2, num_latent))
        noise = Input((res, res, 2))

        noise0 = Lambda(lambda x: x[:, :, :, 0])(noise)
        noise1 = Lambda(lambda x: x[:, :, :, 1])(noise)
        w0 = Lambda(lambda x: x[:, 0])(w)
        w1 = Lambda(lambda x: x[:, 1])(w)

        h = UpSampling2D((2, 2), name='upsampling2d_{0:}x{0:}'.format(res))(x)
        h = ScaledConv2D(
            num_filters,
            (3, 3),
            padding='same',
            use_bias=False,
            use_wscale=use_wscale,
            lr_mul=lr_mul,
            kernel_initializer=get_initializer(
                distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
            name='scaled_conv2d_{0:}x{0:}_0'.format(res))(h)
        h = Blur()(h)
        h = ScaleAdd(name='scale_add_{0:}x{0:}_0'.format(res))([h, noise0])
        h = AddBias2D(name='add_bias2d_{0:}x{0:}_0'.format(res))(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = AdaIN_block(
            res, num_filters, num_latent,
            use_wscale=use_wscale, lr_mul=lr_mul, distribution=distribution)([h, w0])

        h = ScaledConv2D(
            num_filters,
            (3, 3),
            padding='same',
            use_bias=False,
            use_wscale=use_wscale,
            lr_mul=lr_mul,
            kernel_initializer=get_initializer(
                distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
            name='scaled_conv2d_{0:}x{0:}_1'.format(res))(h)
        h = ScaleAdd(name='scale_add_{0:}x{0:}_1'.format(res))([h, noise1])
        h = AddBias2D(name='add_bias2d_{0:}x{0:}_1'.format(res))(h)
        h = LeakyReLU(alpha=0.2)(h)
        y = AdaIN_block(
            res, num_filters, num_latent,
            use_wscale=use_wscale, lr_mul=lr_mul, distribution=distribution)([h, w1])
        return Model([x, w, noise], y)

def toRGB(x_shape, res, num_channels, use_wscale=True, lr_mul=1.0, distribution='untruncated_normal'):
    with tf.name_scope('toRGB_{0:}x{0:}'.format(res)):
        x = Input(x_shape)
        y = ScaledConv2D(
            num_channels,
            (1, 1),
            padding='same',
            use_wscale=use_wscale,
            lr_mul=lr_mul,
            kernel_initializer=get_initializer(
                distribution=distribution, use_wscale=use_wscale),
            name='scaled_conv2d_{0:}x{0:}_toRGB'.format(res))(x)
        return Model(x, y)

class GeneratorSynthesis(Layer):
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
        self.res_out = res_out
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
        res = 4
        with tf.name_scope('generator_synthesis'):
            self.mylayers['image_out'] = [None for _ in range(self.num_blocks)]
            self.mylayers['block'] = [None for _ in range(self.num_blocks)]
            self.mylayers['toRGB'] = [None for _ in range(self.num_blocks)]
            self.mylayers['x_out'] = [None for _ in range(self.num_blocks)]
            self.mylayers['slice_w'] = [None for _ in range(self.num_blocks)]
            self.lod = [None for _ in range(self.num_blocks)]

            self.mylayers['slice_w'][0] = Lambda(lambda x: x[:, :2])
            self.mylayers['const_block'] = const_block(
                res, self.res2num_filters(res), self.num_latent,
                use_wscale=self.use_wscale, lr_mul=self.lr_mul, distribution=self.distribution)
            input_shape = (res, res, self.res2num_filters(res))
            self.mylayers['image_out'][0] = toRGB(
                input_shape, res, self.num_channels,
                use_wscale=self.use_wscale, lr_mul=self.lr_mul, distribution=self.distribution)

            for i in range(1, self.num_blocks):
                input_shape = (res, res, self.res2num_filters(res))
                res *= 2
                self.lod[i] = tf.constant(i, tf.float32)
                self.mylayers['slice_w'][i] = Lambda(lambda x: x[:, i * 2:(i + 1) * 2])
                self.mylayers['block'][i] = generator_block(
                    input_shape, res, self.res2num_filters(res), self.num_latent,
                    use_wscale=self.use_wscale, lr_mul=self.lr_mul, distribution=self.distribution)

                input_shape = (res, res, self.res2num_filters(res))
                self.mylayers['toRGB'][i] = toRGB(
                    input_shape, res, self.num_channels,
                    use_wscale=self.use_wscale, lr_mul=self.lr_mul, distribution=self.distribution)
                self.mylayers['image_out'][i] = UpSampling2D(
                    (2, 2), name='upsampling2d_image_out_{0:}x{0:}'.format(res))
                self.mylayers['x_out'][i] = UpSampling2D(
                    (2, 2), name='upsampling2d_x_out_{0:}x{0:}'.format(res))

    def call(self, inputs, training=None):
        lod, w, *noise = inputs
        lod = tf.reshape(lod, [-1])[0]

        with tf.name_scope('generator_synthesis'):
            w0 = self.mylayers['slice_w'][0](w)
            x = self.mylayers['const_block']([w0, noise[0]])
            image_out = self.mylayers['image_out'][0](x)

            if self.mode == 'static':
                for i in range(1, self.num_blocks):
                    image_out = self.mylayers['image_out'][i](image_out)
                    w_i = self.mylayers['slice_w'][i](w)
                    x = self.mylayers['block'][i]([x, w_i, noise[i]])
                    y = self.mylayers['toRGB'][i](x)
                    image_out = interpolate_clip(y, image_out, self.lod[i] - lod)

            elif self.mode == 'dynamic':
                for i in range(1, self.num_blocks):
                    @tf.function
                    def block_i(x, image_out, w, noise, lod):
                        if self.lod[i] >= lod + 1:
                            x = self.mylayers['x_out'][i](x)
                            image_out = self.mylayers['image_out'][i](image_out)
                        else:
                            w_i = self.mylayers['slice_w'][i](w)
                            x = self.mylayers['block'][i]([x, w_i, noise[i]])
                            if self.lod[i] <= lod:
                                image_out = self.mylayers['toRGB'][i](x)
                            else:
                                x_new = self.mylayers['toRGB'][i](x)
                                image_out_new = self.mylayers['image_out'][i](image_out)
                                image_out = interpolate_clip(
                                    x_new, image_out_new, self.lod[i] - lod)
                        return x, image_out
                    x, image_out = block_i(x, image_out, w, noise, lod)
        return image_out

def GeneratorMapping(res_out=32,
                     num_input_latent=512,
                     num_mapping_layers=8,
                     num_mapping_latent=512,
                     num_output_latent=512,
                     use_wscale=True,
                     lr_mul=1.0,
                     distribution='untruncated_normal',
                     **kwargs):
    num_blocks = res2num_blocks(res_out)
    num_layers = 2 * num_blocks

    with tf.name_scope('generator_mapping'):
        inputs = Input((num_input_latent,))
        x = PixelNormalization()(inputs)
        for i in range(num_mapping_layers):
            num_latent = num_output_latent if i == num_mapping_layers - 1 else num_mapping_latent
            x = ScaledDense(
                num_latent,
                use_wscale=use_wscale,
                lr_mul=lr_mul,
                kernel_initializer=get_initializer(
                    distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
                name='scaled_dense_{:}'.format(i))(x)
            x = LeakyReLU(alpha=0.2)(x)
        outputs = RepeatVector(num_layers)(x)
        return Model(inputs, outputs, **kwargs)

def StyleMixer(res_out=32,
               num_latent=512,
               mixing_prob=0.9,
               latent_avg_beta=0.995,
               truncation_psi=0.7,
               truncation_cutoff=8,
               **kwargs):
    num_blocks = res2num_blocks(res_out)
    num_layers = 2 * num_blocks

    with tf.name_scope('style_mixer'):
        latent1 = Input((num_layers, num_latent,))
        latent2 = Input((num_layers, num_latent,))
        lod = Input((1,))
        lod_tensor = Reshape((1, 1, 1))(lod)

        outputs = MixStyle(
            num_layers,
            mixing_prob=mixing_prob,
            latent_avg_beta=latent_avg_beta,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff)([latent1, latent2, lod_tensor])
        return Model(inputs=[lod, latent1, latent2], outputs=outputs, **kwargs)

#===============================================================================

def down_sample(x, factor=2):
    with tf.name_scope('down_sample'):
        filter = [1 / factor for _ in range(factor)]
        return Blur(filter=filter, normalize=False, stride=factor)(x)

def discriminator_block(x_shape,
                        res,
                        num_filters,
                        use_wscale=True,
                        lr_mul=1.0,
                        distribution='untruncated_normal',
                        batch_std_group_size=4,
                        batch_std_num_features=1,
                        use_sn=False):

    with tf.name_scope('discriminator_block_{0:}x{0:}'.format(res)):
        x = Input(x_shape)
        if res == 4:
            h = BatchStddev()(x)
            if use_sn:
                h = SNConv2D(
                    num_filters[0],
                    (3, 3),
                    padding='same',
                    use_wscale=use_wscale,
                    lr_mul=lr_mul,
                    kernel_initializer=get_initializer(
                        distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
                    name='sn_conv2d_{0:}x{0:}'.format(res))(h)
            else:
                h = ScaledConv2D(
                    num_filters[0],
                    (3, 3),
                    padding='same',
                    use_wscale=use_wscale,
                    lr_mul=lr_mul,
                    kernel_initializer=get_initializer(
                        distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
                    name='scaled_conv2d_{0:}x{0:}'.format(res))(h)
            h = LeakyReLU(alpha=0.2)(h)
            h = Flatten()(h)
            if use_sn:
                h = SNDense(
                    num_filters[1],
                    use_wscale=use_wscale,
                    lr_mul=lr_mul,
                    kernel_initializer=get_initializer(
                        distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
                    name='sn_dense_{0:}x{0:}_0'.format(res))(h)
            else:
                h = ScaledDense(
                    num_filters[1],
                    use_wscale=use_wscale,
                    lr_mul=lr_mul,
                    kernel_initializer=get_initializer(
                        distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
                    name='scaled_dense_{0:}x{0:}_0'.format(res))(h)
            h = LeakyReLU(alpha=0.2)(h)
            if use_sn:
                y = SNDense(
                    1,
                    use_wscale=use_wscale,
                    lr_mul=lr_mul,
                    kernel_initializer=get_initializer(
                        distribution=distribution, use_wscale=use_wscale),
                    name='sn_dense_{0:}x{0:}_1'.format(res))(h)
            else:
                y = ScaledDense(
                    1,
                    use_wscale=use_wscale,
                    lr_mul=lr_mul,
                    kernel_initializer=get_initializer(
                        distribution=distribution, use_wscale=use_wscale),
                    name='scaled_dense_{0:}x{0:}_1'.format(res))(h)
        else:
            if use_sn:
                h = SNConv2D(
                    num_filters[0],
                    (3, 3),
                    padding='same',
                    use_wscale=use_wscale,
                    lr_mul=lr_mul,
                    kernel_initializer=get_initializer(
                        distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
                    name='sn_conv2d_{0:}x{0:}_0'.format(res))(x)
            else:
                h = ScaledConv2D(
                    num_filters[0],
                    (3, 3),
                    padding='same',
                    use_wscale=use_wscale,
                    lr_mul=lr_mul,
                    kernel_initializer=get_initializer(
                        distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
                    name='scaled_conv2d_{0:}x{0:}_0'.format(res))(x)
            h = LeakyReLU(alpha=0.2)(h)
            h = Blur()(h)

            if use_sn:
                h = SNConv2D(
                    num_filters[1],
                    (3, 3),
                    padding='same',
                    use_wscale=use_wscale,
                    lr_mul=lr_mul,
                    kernel_initializer=get_initializer(
                        distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
                    name='sn_conv2d_{0:}x{0:}_1'.format(res))(h)
            else:
                h = ScaledConv2D(
                    num_filters[1],
                    (3, 3),
                    padding='same',
                    use_wscale=use_wscale,
                    lr_mul=lr_mul,
                    kernel_initializer=get_initializer(
                        distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
                    name='scaled_conv2d_{0:}x{0:}_1'.format(res))(h)
            h = down_sample(h, factor=2)
            h = AddBias2D(name='add_bias2d_{0:}x{0:}'.format(res))(h)
            y = LeakyReLU(alpha=0.2)(h)
        return Model(x, y)

def fromRGB(x_shape, res, num_filters, use_wscale=True, lr_mul=1.0,
            distribution='untruncated_normal', use_sn=False):
    with tf.name_scope('fromRGB_{0:}x{0:}'.format(res)):
        x = Input(x_shape)
        if use_sn:
            h = SNConv2D(
                num_filters,
                (1, 1),
                padding='same',
                use_wscale=use_wscale,
                lr_mul=lr_mul,
                kernel_initializer=get_initializer(
                    distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
                name='sn_conv2d_{0:}x{0:}_fromRGB'.format(res))(x)
        else:
            h = ScaledConv2D(
                num_filters,
                (1, 1),
                padding='same',
                use_wscale=use_wscale,
                lr_mul=lr_mul,
                kernel_initializer=get_initializer(
                    distribution=distribution, use_wscale=use_wscale, relu_alpha=0.2),
                name='scale_conv2d_{0:}x{0:}_fromRGB'.format(res))(x)
        y = LeakyReLU(alpha=0.2)(h)
        return Model(x, y)

class Discriminator(Layer):
    def __init__(self, res=32,
                num_channels_in=3,
                num_channels=1,
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
        self.num_channels_in = num_channels_in
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
        self.mylayers = {}
        self._build_layers()

    def res2num_filters(self, res):
        return min(int(self.fmap_base * (2.0 / res) ** self.fmap_decay), self.fmap_max)

    def _build_layers(self):
        res = self.res
        input_shape = (res, res, self.num_channels_in)

        with tf.name_scope('discriminator'):
            self.mylayers['fromRGB'] = [None for _ in range(self.num_blocks)]
            self.mylayers['block'] = [None for _ in range(self.num_blocks)]
            self.lod = [None for _ in range(self.num_blocks)]
            self.mylayers['fromRGB'][0] = fromRGB(
                input_shape, res, self.res2num_filters(res),
                use_wscale=self.use_wscale, lr_mul=self.lr_mul,
                distribution=self.distribution, use_sn=self.use_sn)

            for k in range(1, self.num_blocks):
                i = self.num_blocks - k
                input_shape = (res, res, self.res2num_filters(res))
                self.lod[k] = K.constant(i, tf.float32)
                num_filters = (self.res2num_filters(res), self.res2num_filters(res // 2))
                self.mylayers['block'][k] = discriminator_block(
                    input_shape, res, num_filters,
                    use_wscale=self.use_wscale, lr_mul=self.lr_mul,
                    distribution=self.distribution, use_sn=self.use_sn)

                res = res // 2
                input_shape = (res, res, self.num_channels_in)
                self.mylayers['fromRGB'][k] = fromRGB(
                    input_shape, res, self.res2num_filters(res),
                    use_wscale=self.use_wscale, lr_mul=self.lr_mul,
                    distribution=self.distribution, use_sn=self.use_sn)

            input_shape = (res, res, self.res2num_filters(res))
            num_filters = (self.res2num_filters(res), self.res2num_filters(res // 2))
            self.mylayers['output'] = discriminator_block(
                input_shape, res, num_filters,
                use_wscale=self.use_wscale, lr_mul=self.lr_mul,
                distribution=self.distribution,
                batch_std_group_size=self.batch_std_group_size,
                batch_std_num_features=self.batch_std_num_features,
                use_sn=self.use_sn)

    def call(self, inputs, training=None):
        lod, image = inputs
        lod = tf.reshape(lod, [-1])[0]

        with tf.name_scope('discriminator'):
            x = self.mylayers['fromRGB'][0](image)

            if self.mode == 'static':
                for k in range(1, self.num_blocks):
                    i = self.num_blocks - k
                    image = down_sample(image, factor=2)
                    y = self.mylayers['fromRGB'][k](image)
                    x = self.mylayers['block'][k](x)
                    x = interpolate_clip(x, y, self.lod[k] - lod)

            elif self.mode == 'dynamic':
                for k in range(1, self.num_blocks):
                    i = self.num_blocks - k
                    image = down_sample(image, factor=2)
                    @tf.function
                    def block_i(x, image, lod):
                        if self.lod[k] >= lod + 1:
                            x = self.mylayers['fromRGB'][k](image)
                        elif self.lod[k] <= lod:
                            x = self.mylayers['block'][k](x)
                        else:
                            x_new = self.mylayers['block'][k](x)
                            y_new = self.mylayers['fromRGB'][k](image)
                            x = interpolate_clip(x_new, y_new, self.lod[k] - lod)
                        return x
                    x = block_i(x, image, lod)
            outputs = self.mylayers['output'](x)
        return outputs
