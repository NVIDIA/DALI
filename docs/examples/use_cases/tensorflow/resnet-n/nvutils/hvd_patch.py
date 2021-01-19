from tensorflow import keras
from horovod.tensorflow import Compression
from tensorflow.python.keras.optimizer_v2 import optimizer_v2

import horovod.tensorflow as hvd
import tensorflow as tf
from nvutils import common

def create_distributed_optimizer(keras, optimizer, name, device_dense,
                                 device_sparse, compression, sparse_as_dense):
  class _DistributedOptimizer(optimizer_v2.OptimizerV2):
    _HAS_AGGREGATE_GRAD = True

    def __init__(self, **kwargs):
      self._name = name or "Distributed%s" % self.__class__.__base__.__name__
      self._device_dense = device_dense
      self._device_sparse = device_sparse
      self._compression = compression
      self._sparse_as_dense = sparse_as_dense
      self._aggregated_gradients = False
      super(self.__class__, self).__init__(**kwargs)

    def get_gradients(self, loss, params):
      """
      Compute gradients of all trainable variables.

      See Optimizer.get_gradients() for more info.

      In DistributedOptimizer, get_gradients() is overriden to also
      allreduce the gradients before returning them.
      """
      gradients = super(self.__class__, self).get_gradients(loss, params)
      return self._allreduce(gradients)

    def _aggregate_gradients(self, grads_and_vars):
      gradients = [grad for grad, var in grads_and_vars]
      return self._allreduce(gradients)

    def _allreduce(self, gradients):
      self._aggregated_gradients = True
      if hvd.size() > 1:
        averaged_gradients = []
        with tf.name_scope(self._name + "_Allreduce"):
          for grad in gradients:
            if grad is not None:
              if self._sparse_as_dense and \
                  isinstance(grad, tf.IndexedSlices):
                grad = tf.convert_to_tensor(grad)
              avg_grad = hvd.allreduce(grad,
                                       device_dense=self._device_dense,
                                       device_sparse=self._device_sparse,
                                       compression=self._compression)
              averaged_gradients.append(avg_grad)
            else:
              averaged_gradients.append(None)
          return averaged_gradients
      else:
        return gradients

    def apply_gradients(self, *args, **kwargs):
      if not self._aggregated_gradients:
        raise Exception('`apply_gradients()` was called without a call to '
                        '`get_gradients()` or `_aggregate_gradients`. If '
                        'you\'re using TensorFlow 2.0, please specify '
                        '`experimental_run_tf_function=False` in `compile()`.')
      return super(self.__class__, self).apply_gradients(*args, **kwargs)


  # We dynamically create a new class that inherits from the optimizer that was passed in.
  # The goal is to override get_gradients() method with an allreduce implementation.
  # This class will have the same name as the optimizer it's wrapping, so that the saved
  # model could be easily restored without Horovod.
  cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
             dict(_DistributedOptimizer.__dict__))
  config = optimizer.get_config()
  config['learning_rate'] = \
      common.PiecewiseConstantDecayWithWarmup.from_config(
          config['learning_rate']['config'])
  return cls.from_config(config)

def DistributedOptimizer(optimizer, name=None,
                         device_dense='', device_sparse='',
                         compression=Compression.none,
                         sparse_as_dense=False):
  return create_distributed_optimizer(keras, optimizer, name,
                                      device_dense, device_sparse, compression,
                                      sparse_as_dense)
