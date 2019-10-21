import os
import tensorflow as tf

class BaseModel(object):
    def __init__(self, params, use_tpu=False):
        self.params = params
        self.use_tpu = use_tpu

        tf.keras.backend.clear_session()

        if self.use_tpu:
            tpu_address = os.environ["TPU_NAME"]
            self.cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
            tf.config.experimental_connect_to_cluster(self.cluster_resolver)
            tf.tpu.experimental.initialize_tpu_system(self.cluster_resolver)
            self.strategy = tf.distribute.experimental.TPUStrategy(self.cluster_resolver)
