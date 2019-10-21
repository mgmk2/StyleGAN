import os
import pickle
import numpy as np
import pandas as pd
from ..utils import dataset_utils, image_utils

class ExecutorBase(object):
    def __init__(self, params, use_tpu=False, show_mode='show'):
        self.params = params
        self.use_tpu = use_tpu
        self.show_mode = show_mode

        if self.params.dataset_dir is None:
            self.DATASET_DIR = os.path.join(os.getcwd(), 'dataset')
        else:
            self.DATASET_DIR = self.params.dataset_dir
        self.RESULT_DIR = os.path.join(os.getcwd(), 'result')
        if not os.path.exists(self.RESULT_DIR):
            os.mkdir(self.RESULT_DIR)
        if not os.path.exists(os.path.join(self.RESULT_DIR, 'eval')):
            os.mkdir(os.path.join(self.RESULT_DIR, 'eval'))
        if not os.path.exists(os.path.join(self.RESULT_DIR, 'image')):
            os.mkdir(os.path.join(self.RESULT_DIR, 'image'))

        self.history = {'D loss': [], 'D acc': [],
                        'G loss': [], 'G acc': []}

        self.dataset_train = dataset_utils.get_dataset(self.params.dataset_train)['train']
        assert self.dataset_train['images'].shape[1:] == self.params.image_shape
        if self.params.dataset_eval is None:
            self.dataset_eval = None
        else:
            self.dataset_eval = dataset_utils.get_dataset(self.params.dataset_eval)['test']
            assert self.dataset_eval['images'].shape[1:] == self.params.image_shape

    def _save(self, obj, filename):
        with open(os.path.join(self.RESULT_DIR, filename + '.pkl'), 'wb') as f:
            pickle.dump(obj, f)

    def _load(self, filename):
        with open(os.path.join(self.RESULT_DIR, filename + '.pkl'), 'rb') as f:
            obj = pickle.load(f)
        return obj

    def save_history(self, filename):
        df = pd.DataFrame(self.history)
        df.to_csv(os.path.join(self.RESULT_DIR, filename), index=False)

    def save_images(self, images, filename, epoch=None):
        if epoch is None:
            filename = os.path.join(self.RESULT_DIR, 'eval', filename)
        else:
            filename = os.path.join(self.RESULT_DIR, 'image', filename)
        save_dir = os.path.dirname(filename)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_utils.save_images(images, filename, epoch=epoch)
