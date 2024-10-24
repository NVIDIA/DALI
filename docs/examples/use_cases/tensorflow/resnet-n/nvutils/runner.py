#!/usr/bin/env python
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from nvutils import image_processing
from nvutils import common
from packaging.version import Version

import tensorflow as tf
from tensorflow import keras
import os
import time
import re
import horovod.tensorflow.keras as hvd

from keras import backend
print(tf.__version__)
if Version(tf.__version__) > Version("2.1.0"):
  if Version(tf.__version__) >= Version("2.4.0"):
    from tensorflow.python.keras.mixed_precision import device_compatibility_check
  else:
    from tensorflow.python.keras.mixed_precision.experimental import device_compatibility_check
  device_compatibility_check._logged_compatibility_check = True

class _ProfileKerasFitCallback(keras.callbacks.Callback):
  def __init__(self, batch_size, display_every=10):
    self.batch_size = batch_size * hvd.size()
    self.log_steps = display_every
    self.global_steps = 0

  def on_batch_begin(self, batch, logs=None):
    self.global_steps += 1
    if self.global_steps == 1:
      self.start_time = time.time()

  def on_batch_end(self, batch, logs=None):
    """Records elapse time of the batch and calculates examples per second."""
    if self.global_steps % self.log_steps == 0:
      timestamp = time.time()
      elapsed_time = timestamp - self.start_time
      examples_per_second = (self.batch_size * self.log_steps) / elapsed_time
      if hvd.rank() == 0:
        print("global_step: %d images_per_sec: %.1f" % (self.global_steps,
                                                        examples_per_second))
      self.start_time = timestamp

  def on_epoch_begin(self, epoch, logs=None):
    self.epoch_start = time.time()

  def on_epoch_end(self, epoch, logs=None):
    epoch_run_time = time.time() - self.epoch_start
    if hvd.rank() == 0:
      print("epoch: %d time_taken: %.1f" % (epoch, epoch_run_time))

def train(model_func, params):
  image_width = params['image_width']
  image_height = params['image_height']
  image_format = params['image_format']
  distort_color = params['distort_color']
  momentum = params['momentum']
  loss_scale = params['loss_scale']
  data_dir = params['data_dir']
  data_idx_dir = params['data_idx_dir']
  batch_size = params['batch_size']
  num_iter = params['num_iter']
  iter_unit = params['iter_unit']
  log_dir = params['log_dir']
  export_dir = params['export_dir']
  tensorboard_dir = params['tensorboard_dir']
  display_every = params['display_every']
  precision = params['precision']
  dali_mode = params['dali_mode']
  use_xla = params['use_xla']

  if data_dir is not None:
    file_format = os.path.join(data_dir, '%s-*')
    train_files = sorted(tf.io.gfile.glob(file_format % 'train'))
    valid_files = sorted(tf.io.gfile.glob(file_format % 'validation'))
    num_train_samples = common.get_num_records(train_files)
    num_valid_samples = common.get_num_records(valid_files)
  else:
    num_train_samples = 1281982
    num_valid_samples = 5000

  train_idx_files = None
  valid_idx_files = None
  if data_idx_dir is not None:
    file_format = os.path.join(data_idx_dir, '%s-*')
    train_idx_files = sorted(tf.io.gfile.glob(file_format % 'train'))
    valid_idx_files = sorted(tf.io.gfile.glob(file_format % 'validation'))

  if iter_unit.lower() == 'epoch':
    num_epochs = num_iter
    nstep_per_epoch = num_train_samples // (batch_size * hvd.size())
    nstep_per_valid = num_valid_samples // (batch_size * hvd.size())
  else:
    assert iter_unit.lower() == 'batch'
    num_epochs = 1
    nstep_per_epoch = min(num_iter,
                          num_train_samples // (batch_size * hvd.size()))
    nstep_per_valid = min(10, num_valid_samples // (batch_size * hvd.size()))

  initial_epoch = 0
  if log_dir:
    # We save check points only when using the real data.
    assert data_dir, "--data_dir cannot be empty when using --log_dir"
    assert os.path.exists(log_dir)
    ckpt_format = log_dir +"/model-{epoch:02d}-{val_top1:.2f}.hdf5"
    # Looks for the most recent checkpoint and sets the initial epoch from it.
    for filename in os.listdir(log_dir):
      if filename.startswith('model-'):
        initial_epoch = max(int(re.findall(r'\d+', filename)[0]),
                            initial_epoch)

  if tensorboard_dir:
    assert os.path.exists(tensorboard_dir)

  if export_dir:
    assert os.path.exists(export_dir)
    save_format = export_dir +"/saved_model_rn50.h5"

  if use_xla:
    tf.config.optimizer.set_jit(True)

  # Horovod: pin GPU to be used to process local rank (one GPU per process)
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

  if precision == 'fp16':
    if Version(tf.__version__) >= Version("2.4.0"):
      policy = keras.mixed_precision.Policy('mixed_float16')
      keras.mixed_precision.set_global_policy(policy)
    else:
      policy = keras.mixed_precision.experimental.Policy('mixed_float16', loss_scale)
      keras.mixed_precision.experimental.set_policy(policy)

  lr_schedule = common.create_piecewise_constant_decay_with_warmup(
      batch_size=batch_size * hvd.size(),
      epoch_size=num_train_samples,
      warmup_epochs=common.LR_SCHEDULE[0][1],
      boundaries=list(p[1] for p in common.LR_SCHEDULE[1:]),
      multipliers=list(p[0] for p in common.LR_SCHEDULE),
      compute_lr_on_cpu=True)
  opt = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=momentum)
  # Horovod: add Horovod DistributedOptimizer. We use a modified version to
  # support the custom learning rate schedule.
  opt = hvd.DistributedOptimizer(opt)
  if Version(tf.__version__) >= Version("2.4.0") and precision == 'fp16':
    opt = keras.mixed_precision.LossScaleOptimizer(opt, dynamic=False,
                                                   initial_scale=loss_scale)

  backend.set_image_data_format(image_format)
  dtype='float16' if precision == 'fp16' else 'float32'
  backend.set_floatx(dtype)
  model = model_func(num_classes=image_processing.NUM_CLASSES)
  loss_func ='sparse_categorical_crossentropy',

  top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5')
  top1 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='top1')

  # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
  # uses hvd.DistributedOptimizer() to compute gradients. However, this option
  # will disable the overlapping of the data loading and compute and hurt the
  # performace if the model is not under the scope of distribution strategy
  # scope.
  model.compile(optimizer=opt, loss=loss_func, metrics=[top1, top5],
                experimental_run_tf_function=False)

  training_hooks = []
  training_hooks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
  training_hooks.append(_ProfileKerasFitCallback(batch_size, display_every))

  if log_dir and hvd.rank() == 0:
    ckpt_callback = keras.callbacks.ModelCheckpoint(ckpt_format,
        monitor='val_top1', verbose=1, save_best_only=False,
        save_weights_only=False, save_frequency=1)
    training_hooks.append(ckpt_callback)

  if tensorboard_dir and hvd.rank() == 0:
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_dir)
    training_hooks.append(tensorboard_callback)

  if data_dir is not None:
    num_preproc_threads = params['dali_threads'] if dali_mode else 10
    train_input = image_processing.image_set(train_files, batch_size,
        image_height, image_width, training=True, distort_color=distort_color,
        deterministic=False, num_threads=num_preproc_threads,
        use_dali=dali_mode, idx_filenames=train_idx_files)

    valid_input = image_processing.image_set(valid_files, batch_size,
        image_height, image_width, training=False, distort_color=False,
        deterministic=False, num_threads=num_preproc_threads,
        use_dali=dali_mode, idx_filenames=valid_idx_files)
    if dali_mode:
      train_input = train_input.get_device_dataset()
      valid_input = valid_input.get_device_dataset()
    valid_params = {'validation_data': valid_input,
                    'validation_steps': nstep_per_valid,
                    'validation_freq': 1}
  else:
    train_input = image_processing.fake_image_set(batch_size, image_height,
                                                  image_width)
    valid_params = {}

  try:
    verbose = 2 if hvd.rank() == 0 else 0
    model.fit(train_input, epochs=num_epochs, callbacks=training_hooks,
              steps_per_epoch=nstep_per_epoch, verbose=verbose,
              initial_epoch=initial_epoch, **valid_params)
  except KeyboardInterrupt:
    print("Keyboard interrupt")

  if export_dir and hvd.rank() == 0:
    model.save(save_format)
    print(f"The model is saved to {save_format}")

def predict(params):
  image_width = params['image_width']
  image_height = params['image_height']
  batch_size = params['batch_size']
  export_dir = params['export_dir']

  assert export_dir, "--export_dir must be given."
  model_path = export_dir +"/saved_model_rn50.h5"
  assert os.path.exists(model_path)

  model = keras.models.load_model(model_path, custom_objects={
      "PiecewiseConstantDecayWithWarmup":
          common.PiecewiseConstantDecayWithWarmup})

  predict_input = image_processing.fake_image_set(batch_size, image_height,
                                                  image_width, with_label=False)
  results = model.predict(predict_input, verbose=1, steps=3)
  print(f"The loaded model predicts {results.shape[0]} images.")

