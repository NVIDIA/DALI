import tensorflow as tf
from packaging.version import Version

BASE_LEARNING_RATE = 0.1

LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]

def create_piecewise_constant_decay_with_warmup(batch_size, epoch_size,
    warmup_epochs, boundaries, multipliers, compute_lr_on_cpu=True, name=None):
  if len(boundaries) != len(multipliers) - 1:
    raise ValueError('The length of boundaries must be 1 less than the '
                     'length of multipliers')
  base_lr_batch_size = 256
  steps_per_epoch = epoch_size // batch_size

  rescaled_lr = BASE_LEARNING_RATE * batch_size / base_lr_batch_size
  step_boundaries = [float(steps_per_epoch) * x for x in boundaries]
  lr_values = [rescaled_lr * m for m in multipliers]
  if Version(tf.__version__) >= Version("2.13"):
    warmup_steps = int(warmup_epochs * steps_per_epoch)
  else:
    warmup_steps = warmup_epochs * steps_per_epoch
  compute_lr_on_cpu = compute_lr_on_cpu
  name = name
  return PiecewiseConstantDecayWithWarmup(rescaled_lr, step_boundaries,
                                          lr_values, warmup_steps,
                                          compute_lr_on_cpu, name)

@tf.keras.utils.register_keras_serializable(package='Custom')
class PiecewiseConstantDecayWithWarmup(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  """Piecewise constant decay with warmup schedule."""

  def __init__(self, rescaled_lr, step_boundaries, lr_values, warmup_steps,
               compute_lr_on_cpu, name):
    super(PiecewiseConstantDecayWithWarmup, self).__init__()

    self.rescaled_lr = rescaled_lr
    if Version(tf.__version__) >= Version("2.13"):
      self.step_boundaries = [int(b) for b in step_boundaries]
    else:
      self.step_boundaries = step_boundaries
    self.lr_values = lr_values
    self.warmup_steps = warmup_steps
    self.compute_lr_on_cpu = compute_lr_on_cpu
    self.name = name

    self.learning_rate_ops_cache = {}

  def __call__(self, step):
    if tf.executing_eagerly():
      return self._get_learning_rate(step)

    # In an eager function or graph, the current implementation of optimizer
    # repeatedly call and thus create ops for the learning rate schedule. To
    # avoid this, we cache the ops if not executing eagerly.
    graph = tf.compat.v1.get_default_graph()
    if graph not in self.learning_rate_ops_cache:
      if self.compute_lr_on_cpu:
        with tf.device('/device:CPU:0'):
          self.learning_rate_ops_cache[graph] = self._get_learning_rate(step)
      else:
        self.learning_rate_ops_cache[graph] = self._get_learning_rate(step)
    return self.learning_rate_ops_cache[graph]

  def _get_learning_rate(self, step):
    """Compute learning rate at given step."""
    with tf.compat.v1.name_scope(self.name, 'PiecewiseConstantDecayWithWarmup',
                                 [self.rescaled_lr, self.step_boundaries,
                                  self.lr_values, self.warmup_steps,
                                  self.compute_lr_on_cpu]):
      def warmup_lr(step):
        return self.rescaled_lr * (
            tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32))
      def piecewise_lr(step):
        return tf.compat.v1.train.piecewise_constant(
            step, self.step_boundaries, self.lr_values)
      return tf.cond(step < self.warmup_steps,
                     lambda: warmup_lr(step),
                     lambda: piecewise_lr(step))

  def get_config(self):
    return {
        'rescaled_lr': self.rescaled_lr,
        'step_boundaries': self.step_boundaries,
        'lr_values': self.lr_values,
        'warmup_steps': self.warmup_steps,
        'compute_lr_on_cpu': self.compute_lr_on_cpu,
        'name': self.name
    }

def get_num_records(filenames):
  def count_records(tf_record_filename):
    count = 0
    for _ in tf.compat.v1.python_io.tf_record_iterator(tf_record_filename):
      count += 1
    return count
  nfile = len(filenames)
  return (count_records(filenames[0])*(nfile-1) +
          count_records(filenames[-1]))

