import tensorflow as tf

def tpu_decorator(func):
    def wrapper(self, *args, **kwargs):
        if self.use_tpu and not tf.distribute.in_cross_replica_context():
            with self.strategy.scope():
                output = func(self, *args, **kwargs)
        else:
            output = func(self, *args, **kwargs)
        return output
    return wrapper

def tpu_ops_decorator(mode):
    def _tpu_ops_decorator(func):
        def wrapper(self, *args, **kwargs):
            outputs = []

            if self.use_tpu:
                _func = lambda *args, **kwargs: func(self, *args, **kwargs)
                func_outputs = self.strategy.experimental_run_v2(_func, args=args, kwargs=kwargs)
            else:
                func_outputs = func(self, *args, **kwargs)
            if not isinstance(func_outputs, tuple):
                func_outputs = (func_outputs,)
            for x in func_outputs:
                if isinstance(mode, str) and mode.upper() == 'SUM':
                    if self.use_tpu:
                        y = self.strategy.reduce(
                            tf.distribute.ReduceOp.SUM, x, axis=None)
                    else:
                        y = tf.reduce_sum(x)
                else:
                    if self.use_tpu:
                        y = tf.concat(x.values, axis=0)
                    else:
                        y = x
                outputs.append(y)
            if len(outputs) > 1:
                return tuple(outputs)
            elif len(outputs) == 1:
                return outputs[0]
        return wrapper
    return _tpu_ops_decorator

def convert_to_tfdata_single_batch(func):
    def wrapper(self, inputs, *args, **kwargs):
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        dataset = dataset.batch(self.params.batch_size, drop_remainder=True)
        if self.use_tpu:
            dataset = self.strategy.experimental_distribute_dataset(dataset)
        outputs = func(self, next(iter(dataset)), *args, **kwargs)
        return outputs
    return wrapper
