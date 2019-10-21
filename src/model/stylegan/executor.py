import os
import sys
import time
import numpy as np
from .network import res2num_blocks
from .model import StyleGANModel
from ..base_executor import ExecutorBase
from ...utils import image_utils, utils
from ...utils.decorator import tpu_decorator

class StyleGAN(ExecutorBase):
    def __init__(self, params, use_tpu=False, mode=None, show_mode='show'):
        super().__init__(params, use_tpu=use_tpu, show_mode=show_mode)
        self.N_train = self.dataset_train['images'].shape[0]
        self.N_batches = self.N_train // self.params.batch_size
        self.params.lr_schedule = self.get_lr_schedule()

        self.model = StyleGANModel(params, use_tpu=use_tpu, mode=mode)
        if self.use_tpu:
            self.strategy = self.model.strategy

        print('number of batches:', self.N_batches)
        print('Level of details range: 0.0 - {:.1f}'.format(self.get_maximum_lod()))

    def get_lr_schedule(self):
        if hasattr(self.params, 'lr_schedule'):
            return self.params.lr_schedule

        steps_at_lod_period = self.params.epochs_at_lod_period * self.N_batches
        steps_for_progressive = self.params.epochs_for_progressive * self.N_batches

        steps = steps_at_lod_period
        boundaries = []
        values = [self.params.learning_rate * self.params.lr_mul_shedule[0]]
        for lod in range(1, self.get_maximum_lod() + 1):
            if lod in self.params.lr_mul_shedule.keys():
                if len(boundaries) == 0:
                    boundaries.append(steps - 1)
                else:
                    boundaries.append(steps)
                values.append(
                    self.params.learning_rate * self.params.lr_mul_shedule[lod])
            steps += steps_at_lod_period + steps_for_progressive
        return {'boundaries': boundaries, 'values': values}

    def save_weights(self, filename, epoch):
        weights = {k: v.numpy() for k, v in self.model.get_weights().items()}
        obj = {'epoch': epoch, 'weights': weights}
        self._save(obj, filename)

    def load_weights(self, filename, load_optimizer=True, load_epoch=True):
        weights = self._load(filename)
        print('Load weights from ' + filename + '.pkl...')
        if load_epoch:
            self.params.start_epoch = weights['epoch']
        self.model.set_weights(weights['weights'], load_optimizer=load_optimizer)

    def get_maximum_lod(self):
        res = self.params.image_shape[0]
        return utils.num_div2(res) - 2

    def get_z(self, N=None):
        if N is None:
            N = self.N_train
        shape = (N, self.params.z_dim)
        return np.random.normal(0, 1, shape).astype(np.float32)

    def get_noises(self, N=None):
        if N is None:
            N = self.N_train
        assert self.params.image_shape[0] == self.params.image_shape[1]
        res = self.params.image_shape[0]
        num_blocks = res2num_blocks(res)
        noises = [None for _ in range(num_blocks)]
        for i in range(num_blocks):
            res_i = 2 ** (i + 2)
            shape = (N, res_i, res_i, 2)
            noises[i] = np.random.normal(0, 1, shape).astype(np.float32)
        return noises

    def convert_lod(self, lod):
        if self.use_tpu:
            lod = np.array([lod], dtype=np.float32).repeat((8))
        else:
            lod = np.array([lod], dtype=np.float32)
        return utils.convert_to_tensor(lod)

    def show_sample_images(self, lod, epoch=None, mode=None):
        if mode is None:
            mode = self.show_mode
        z = self.get_z(self.params.batch_size)
        noises = self.get_noises(self.params.batch_size)
        images_uint8 = self.dataset_train['images'][:self.params.batch_size]
        images = image_utils.convert_color_range(
            images_uint8, input_range=(0, 255), output_range=(-1, 1))

        lod_input = self.convert_lod(lod)
        inputs = utils.convert_to_tensor((z, images, *noises))
        images_gen_raw = self.model.eval_gen(inputs, lod_input).numpy()
        images_gen = image_utils.convert_color_range(
            images_gen_raw, input_range=(-1, 1), output_range=(0, 1))
        image_utils.show_images(images_gen, epoch=epoch, mode=mode)
        self.save_images(images_gen, 'sample_image.png', epoch=epoch)

    @tpu_decorator
    def eval(self, N=None, lod=None, mode=None):
        if lod is None:
            lod = self.get_maximum_lod()
        if self.use_tpu and N is not None:
            raise ValueError("You can't specify N on TPU.")
        if N is None:
            N = self.params.batch_size

        z = self.get_z(N)
        noises = self.get_noises(N)
        images_uint8 = self.dataset_train['images'][:N]
        images = image_utils.convert_color_range(
            images_uint8, input_range=(0, 255), output_range=(-1, 1))

        lod_input = self.convert_lod(lod)
        inputs = utils.convert_to_tensor((z, images, *noises))
        images_gen_raw = self.model.eval_gen(inputs, lod_input).numpy()
        images_gen = image_utils.convert_color_range(
            images_gen_raw, input_range=(-1, 1), output_range=(0, 1))
        image_utils.show_images(images_gen, mode=mode)
        self.save_images(images_gen, 'eval_image.png')


    @tpu_decorator
    def fit(self,
            iter_ratio=(1, 1),
            show_sample_period=10,
            lod=None,
            reset_opt_for_new_lod=True):
        if lod is None:
            lod = (0, self.get_maximum_lod())
        if isinstance(lod, int) or isinstance(lod, float):
            lod = (lod, lod)

        if lod[0] == lod[1]:
            if lod[0] == self.get_maximum_lod():
                epochs = self.params.epochs_at_lod_max
            else:
                epochs = self.params.epochs_at_lod_period
        elif lod[0] == 0:
            epochs = self.params.epochs_at_lod_period * (lod[1] - lod[0] + 1) + \
                     self.params.epochs_for_progressive * (lod[1] - lod[0])
        else:
            epochs = self.params.epochs_at_lod_period * (lod[1] - lod[0]) + \
                     self.params.epochs_for_progressive * (lod[1] - lod[0])
        epochs = int(epochs)

        lod_epoch = lod[0]
        prev_lod_epoch = lod[0]

        self.show_sample_images(lod=lod[0], epoch=self.params.start_epoch)

        indices = np.arange(self.N_train)
        end_epoch = self.params.start_epoch + epochs
        for epoch in range(self.params.start_epoch, end_epoch):
            time_start_epoch = time.time()
            d_loss_epoch = 0
            g_loss_epoch = 0

            epochs_period = int(lod_epoch - lod[0]) * (
                self.params.epochs_for_progressive + self.params.epochs_at_lod_period)
            epochs_after_period = (epoch - self.params.start_epoch) - epochs_period
            if lod[0] == 0:
                epochs_after_period = max(
                    -1, epochs_after_period - self.params.epochs_at_lod_period)
            if lod_epoch < lod[1] and \
               epochs_after_period <= self.params.epochs_for_progressive:
                progressive_epochs = epochs_after_period + 1
                lod_epoch = int(lod_epoch) + \
                    progressive_epochs / self.params.epochs_for_progressive

            if reset_opt_for_new_lod:
                if lod != self.get_maximum_lod() \
                    and np.ceil(lod_epoch) != np.ceil(prev_lod_epoch):

                    print('Reset optimizer state.')
                    self.model.initialize_optimizer()

            lod_input = self.convert_lod(lod_epoch)
            np.random.shuffle(indices)
            for iteration in range(self.N_batches):
                z = self.get_z(self.params.batch_size)
                noises = self.get_noises(self.params.batch_size)
                indices_epoch = \
                    indices[iteration * self.params.batch_size:(iteration + 1) * self.params.batch_size]
                images_batch_uint8 = self.dataset_train['images'][indices_epoch]
                images = image_utils.convert_color_range(
                    images_batch_uint8, input_range=(0, 255), output_range=(-1, 1))
                inputs = utils.convert_to_tensor((z, images, *noises))

                if iteration % iter_ratio[0] == 0:
                    d_loss = self.model.train_disc(inputs, lod_input)

                if iteration % iter_ratio[1] == 0:
                    g_loss = self.model.train_gen(inputs, lod_input)

                d_loss_epoch += d_loss
                g_loss_epoch += g_loss

                sys.stdout.write(
                    ('\repoch:{:d}  iter:{:d}  lod:{:.2f}  '
                     '[D loss: {:f}] [G loss: {:f}]   ').format(
                        epoch + 1, iteration + 1, lod_epoch,
                        d_loss, g_loss))
                sys.stdout.flush()

            d_loss_epoch /= self.N_batches
            g_loss_epoch /= self.N_batches
            self.history['D loss'].append(d_loss_epoch)
            self.history['D acc'].append(None)
            self.history['G loss'].append(g_loss_epoch)
            self.history['G acc'].append(None)

            epoch_time = time.time() - time_start_epoch
            sys.stdout.write(
                ('\repoch:{:d}  iter:{:d}  lod:{:.2f}  '
                 '[D loss: {:f}] [G loss: {:f}]   time: {:f}\n').format(
                    epoch + 1, iteration + 1, lod_epoch,
                    d_loss_epoch, g_loss_epoch, epoch_time))
            sys.stdout.flush()

            if (epoch + 1) % show_sample_period == 0:
                self.show_sample_images(lod=lod_epoch, epoch=epoch + 1)
            prev_lod_epoch = lod_epoch

        self.params.start_epoch += epochs
