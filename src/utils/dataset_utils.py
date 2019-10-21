import numpy as np
import tensorflow as tf
from ..dataset import flickr_face

def get_dataset(dataset_name, *args, **kwargs):
    dataset_name = dataset_name.lower()
    if dataset_name == 'cifar10':
        datasets_train, datasets_test = \
            tf.keras.datasets.cifar10.load_data(*args, **kwargs)
    elif dataset_name == 'cifar100':
        datasets_train, datasets_test = \
            tf.keras.datasets.cifar100.load_data(*args, **kwargs)
    elif dataset_name == 'mnist':
        datasets_train, datasets_test = \
            tf.keras.datasets.mnist.load_data(*args, **kwargs)
        datasets_train = (datasets_train[0][..., np.newaxis], datasets_train[1])
        datasets_test = (datasets_test[0][..., np.newaxis], datasets_test[1])
    elif dataset_name == 'fashion_mnist':
        datasets_train, datasets_test = \
            tf.keras.datasets.fashion_mnist.load_data(*args, **kwargs)
    elif 'flickr_face' in dataset_name:
        s = dataset_name.replace('flickr_face', '')
        if 'train' in s:
            dataset_type = 'train'
            res_str = s.replace('_train', '')
        elif 'test' in s:
            dataset_type = 'test'
            res_str = s.replace('_test', '')
        else:
            dataset_type = None
            res_str = s
        res = int(res_str)
        datasets_train, datasets_test = \
            flickr_face.load_data(res, *args, dataset_type=dataset_type, **kwargs)
    else:
        raise ValueError('Unknown dataset name: ' + dataset_name)
    return {'train': tuple_to_dict(datasets_train),
            'test': tuple_to_dict(datasets_test)}

def tuple_to_dict(datasets):
    return {'images': datasets[0], 'labels': datasets[1]}
