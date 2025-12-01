# This is a patch for Horovod 0.21.3 to work with our custom learning schedule
# used in CNN resnet50 scripts.

from tensorflow import keras
from horovod.tensorflow import Compression
from tensorflow.python.keras.optimizer_v2 import optimizer_v2

import horovod.tensorflow as hvd
import tensorflow as tf
from nvutils import common
from packaging.version import Version
from horovod.tensorflow import Average, Compression, Sum

_PRE_TF_2_4_0 = Version(tf.__version__) < Version('2.4.0')

def create_distributed_optimizer(
        keras, optimizer, name, device_dense, device_sparse, compression,
        sparse_as_dense, gradient_predivide_factor, op,
        backward_passes_per_step=1, average_aggregated_gradients=False,
        groups=None):
  class _DistributedOptimizer(keras.optimizers.Optimizer):
    _HAS_AGGREGATE_GRAD = True

    def __init__(self, **kwargs):
      self._name = name or "Distributed%s" % self.__class__.__base__.__name__
      self._aggregated_gradients = False

      self._allreduce_grads = hvd._make_allreduce_grads_fn(
          self._name,
          device_dense,
          device_sparse,
          compression,
          sparse_as_dense,
          op,
          gradient_predivide_factor,
          groups)

      self._agg_helper = None
      if backward_passes_per_step > 1:
        if hvd._executing_eagerly():
          self._agg_helper = LocalGradientAggregationHelperEager(
              backward_passes_per_step=backward_passes_per_step,
              allreduce_func=self._allreduce_grads,
              sparse_as_dense=sparse_as_dense,
              average_aggregated_gradients=average_aggregated_gradients,
          )
        else:
          self._agg_helper = LocalGradientAggregationHelper(
              backward_passes_per_step=backward_passes_per_step,
              allreduce_func=self._allreduce_grads,
              sparse_as_dense=sparse_as_dense,
              average_aggregated_gradients=average_aggregated_gradients,
              rank=rank(),
              optimizer_type=(
                  LocalGradientAggregationHelper._OPTIMIZER_TYPE_KERAS),
          )

      super(self.__class__, self).__init__(**kwargs)

    def _compute_gradients(self, loss, var_list, grad_loss=None, tape=None):
      """
      Compute gradients of all trainable variables.
      See Optimizer.get_gradients() for more info.
      In DistributedOptimizer, get_gradients() is overriden to also
      allreduce the gradients before returning them.
      """
      if _PRE_TF_2_4_0:
        return super(self.__class__, self)._compute_gradients(
            loss, var_list, grad_loss, tape)

      tape = backprop.GradientTape() if tape is None else tape
      grads_and_vars = super(self.__class__, self)._compute_gradients(
          # pylint: disable=protected-access
          loss,
          var_list,
          grad_loss,
          tape=tape)
      grads, weights = list(zip(*grads_and_vars))

      allreduced_grads = self._allreduce(grads, weights)
      return list(zip(allreduced_grads, weights))

    def get_gradients(self, loss, params):
      """
      Compute gradients of all trainable variables.
      See Optimizer.get_gradients() for more info.
      In DistributedOptimizer, get_gradients() is overriden to also
      allreduce the gradients before returning them.
      """
      gradients = super(self.__class__, self).get_gradients(loss, params)
      return self._allreduce(gradients, params)

    def _aggregate_gradients(self, grads_and_vars):
      if _PRE_TF_2_4_0:
        grads, vars = list(zip(*grads_and_vars))
        aggregated_grads = self._allreduce(grads, vars)
        return aggregated_grads
      else:
        return super(self.__class__, self)._aggregate_gradients(
            grads_and_vars)

    def _allreduce(self, grads, vars):
      self._aggregated_gradients = True

      if self._agg_helper:
        return self._agg_helper.compute_gradients(tuple(grads), tuple(vars))
      else:
        return self._allreduce_grads(grads, vars)

    def apply_gradients(self, *args, **kwargs):
      if self._agg_helper:
        if isinstance(args[0], zip):
          # If grad_and_vars are passed in as a zip object
          # convert to a list. This is necessary for TF2.4+
          # b/c args[0] is used in both conditional branches
          # inside _agg_helper.apply_gradients().
          args = list(args)
          args[0] = list(args[0])
          args = tuple(args)

        results = self._agg_helper.apply_gradients(
            lambda: super(self.__class__, self).apply_gradients(*args, **kwargs),
            self,
            *args,
            **kwargs,
        )
      else:
        results = super(self.__class__, self).apply_gradients(*args, **kwargs)

      if _PRE_TF_2_4_0 and not self._aggregated_gradients:
        raise Exception('`apply_gradients()` was called without a call to '
                        '`get_gradients()` or `_aggregate_gradients`. If '
                        'you\'re using TensorFlow 2.0, please specify '
                        '`experimental_run_tf_function=False` in `compile()`.')

      return results

  # We dynamically create a new class that inherits from the optimizer that was
  # passed in. The goal is to override get_gradients() method with an allreduce
  # implementation. This class will have the same name as the optimizer it's
  # wrapping, so that the saved model could be easily restored without Horovod.
  cls = type(optimizer.__class__.__name__, (optimizer.__class__,),
             dict(_DistributedOptimizer.__dict__))

  # This is the patch to allow the hovorod DistributedOptimizer recognize the
  # custom learning rate schedule we have used in CNN resnet50 scripts.
  config = optimizer.get_config()
  config['learning_rate'] = \
      common.PiecewiseConstantDecayWithWarmup.from_config(
          config['learning_rate']['config'])

  return cls.from_config(config)

def DistributedOptimizer(optimizer, name=None,
                         device_dense='', device_sparse='',
                         compression=Compression.none,
                         sparse_as_dense=False,
                         gradient_predivide_factor=1.0,
                         op=Average,
                         backward_passes_per_step=1,
                         average_aggregated_gradients=False):
  """
  An optimizer that wraps another keras.optimizers.Optimizer, using an allreduce
  to average gradient values before applying gradients to model weights.
  Args:
      optimizer: Optimizer to use for computing gradients and applying updates.
      name: Optional name prefix for the operations created when applying
            gradients. Defaults to "Distributed" followed by the provided
            optimizer type.
      device_dense: Device to be used for dense tensors. Uses GPU by default
                    if Horovod was build with HOROVOD_GPU_OPERATIONS.
      device_sparse: Device to be used for sparse tensors. Uses GPU by default
                     if Horovod was build with HOROVOD_GPU_OPERATIONS.
      compression: Compression algorithm used to reduce the amount of data
                   sent and received by each worker node.  Defaults to not
                   using compression.
      sparse_as_dense: Treat all sparse gradients as dense tensors.  This can
                       help improve performance and memory utilization if
                       the original sparse gradient has high density.
                       Defaults to false.
      gradient_predivide_factor: gradient_predivide_factor splits the averaging
                                 before and after the sum. Gradients are scaled
                                 by 1.0 / gradient_predivide_factor before the
                                 sum and gradient_predivide_factor / size after
                                 the sum.
      op: The reduction operation to use when combining gradients across
          different ranks. Defaults to Average.
      backward_passes_per_step: Number of backward passes to perform before
                                calling hvd.allreduce. This allows accumulating
                                updates over multiple mini-batches before
                                reducing and applying them.
      average_aggregated_gradients: Whether to average the aggregated gradients
                                    that have been accumulated over multiple
                                    mini-batches. If true divides gradient
                                    updates by backward_passes_per_step.
                                    Only applicable for backward_passes_per_step
                                    > 1.
  """
  if gradient_predivide_factor != 1.0 and rocm_built():
    raise ValueError('gradient_predivide_factor not supported yet with ROCm')

  if op != Average and op != Sum:
    raise ValueError('op currently only supports Average and Sum')

  return create_distributed_optimizer(
      keras=keras,
      optimizer=optimizer,
      name=name,
      device_dense=device_dense,
      device_sparse=device_sparse,
      compression=compression,
      sparse_as_dense=sparse_as_dense,
      gradient_predivide_factor=gradient_predivide_factor,
      op=op,
      backward_passes_per_step=backward_passes_per_step,
      average_aggregated_gradients=average_aggregated_gradients,
  )
