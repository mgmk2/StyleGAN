import os, sys
import json
import shutil
import numpy as np
from PIL import Image

def load_data(res, dataset_type=None):
    if res is None:
        res = 128
    with open('dataset.json', 'r') as f:
        d = json.load(f)
    dataset_dir = d['flickr_face']

    if len(dataset_dir) == 0:
        raise ValueError(
            "Not defined directory of Flickr-Faces-HQ Dataset at 'dataset.json'")
    elif not os.path.exists(dataset_dir):
        raise FileNotFoundError("No such directory: '" + dataset_dir + "'")

    if res <= 128:
        dataset_dir = os.path.join(dataset_dir, 'thumbnails128x128')
        resize_flag = (res < 128)
    elif 128 < res <= 1024:
        dataset_dir = os.path.join(dataset_dir, 'images1024x1024')
        resize_flag = (res < 1024)
    else:
        raise ValueError('Resolution of flickr face dataset must be equal or less than 1024.')

    x_train = None
    x_test = None

    if dataset_type is None:
        x = read_images(
            dataset_dir, res, resize_flag, idx_range=(0, 70000))
        x_train, x_test = x, x
    elif dataset_type == 'train':
        x_train = read_images(
            dataset_dir, res, resize_flag, idx_range=(0, 60000))
    elif dataset_type == 'test':
        x_test = read_images(
            dataset_dir, res, resize_flag, idx_range=(60000, 70000))

    return (x_train, None), (x_test, None)

def read_images(dataset_dir, res, resize_flag=True, idx_range=None):
    if idx_range is None:
        idx_range = (0, 70000)
    N = idx_range[1] - idx_range[0]
    x = np.zeros([N, res, res, 3], dtype=np.uint8)

    for i in range(idx_range[0], idx_range[1], 1000):
        part_dir = format(i, '05d')
        for j in range(1000):
            idx = i + j
            filename = '{:05d}.png'.format(idx)
            image_pil = Image.open(os.path.join(dataset_dir, part_dir, filename))
            if resize_flag:
                image_pil = image_pil.resize((res, res), Image.BICUBIC)
            x[idx - idx_range[0]] = np.array(image_pil)
            show_progress(idx, idx_range)
    print()
    return x

def show_progress(idx, idx_range):
    str_before = 'Load Flickr-Faces-HQ Dataset ['
    str_after = ']'
    columns = shutil.get_terminal_size().columns
    bar_length = columns - (len(str_before) + len(str_after)) - 15
    percent = (idx - idx_range[0]) / (idx_range[1] - idx_range[0])
    progress = round(percent * bar_length)
    sys.stdout.write(
        ('\r' + str_before + '=' * progress +
         '-' * (bar_length - progress) + str_after +
         '{: 4d}% '.format(round(percent * 100))))
    sys.stdout.flush()
