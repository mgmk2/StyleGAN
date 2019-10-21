import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def show_images(images, epoch=None, mode='show'):
    if mode not in ['show', 'pause']:
        raise ValueError('Unknown mode to show images: ' + mode)

    if images.shape[-1] == 1:
        x = images[:, :, :, 0]
        cmap = 'gray'
    else:
        x = images
        cmap = None
    fig = plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(x[i].clip(0, 1), cmap=cmap)
        plt.axis('off')
    if epoch is not None:
        fig.suptitle('epoch: {:}'.format(epoch))

    if mode == 'pause':
        plt.pause(.05)
    else:
        plt.show()

def _save_image(image, filename):
    if image.dtype in [np.float16, np.float32, np.float64]:
        image = convert_color_range(
            image.clip(0, 1), input_range=(0, 1), output_range=(0, 255))
        image_uint = (image + 0.5).astype(np.uint8)
    else:
        image_uint = image.astype(np.uint8)

    Image.fromarray(image_uint).save(filename)

def save_images(images, filename, epoch=None):
    if epoch is not None:
        save_dir, filename = os.path.split(filename)
        save_dir = os.path.join(save_dir, '{:}epoch'.format(epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filebase, ext = os.path.splitext(filename)
        filename = filebase + '_' + str(epoch) + ext
        filename = os.path.join(save_dir, filename)

    if isinstance(images, list) or isinstance(images, tuple):
        filebase, ext = os.path.splitext(filename)
        for i, im in enumerate(images):
            filename = filebase + '_{:}'.format(i + 1) + ext
            _save_image(im, filename)

    elif isinstance(images, np.ndarray) and images.ndim > 3:
        rows, cols, ch = images.shape[-3:]
        images = images.reshape((-1, rows, cols, ch))
        filebase, ext = os.path.splitext(filename)
        for i, im in enumerate(images):
            filename = filebase + '_{:}'.format(i + 1) + ext
            _save_image(im, filename)

    else:
        _save_image(images, filename)

def convert_color_range(images,
                        input_range=(-1, 1),
                        output_range=(0, 1),
                        dtype=np.float32):
    input_scale = input_range[1] - input_range[0]
    input_bias = input_range[0]
    output_scale = output_range[1] - output_range[0]
    output_bias = output_range[0]
    images_normalized = (images.astype(dtype) - input_bias) / input_scale
    return images_normalized * output_scale + output_bias
