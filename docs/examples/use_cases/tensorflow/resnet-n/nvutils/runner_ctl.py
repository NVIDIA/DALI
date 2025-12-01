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

from builtins import range
from nvutils import image_processing
from nvutils import common
from packaging.version import Version

import tensorflow as tf
from tensorflow import keras
import os
import time
import re

from keras import backend
print(tf.__version__)
if Version(tf.__version__) > Version("2.1.0"):
  if Version(tf.__version__) >= Version("2.4.0"):
    from tensorflow.python.keras.mixed_precision import device_compatibility_check
  else:
    from tensorflow.python.keras.mixed_precision.experimental import device_compatibility_check
  device_compatibility_check._logged_compatibility_check = True

import horovod.tensorflow as hvd

def train_ctl(model_func, params):
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

  if tensorboard_dir and hvd.rank() == 0:
    assert os.path.exists(tensorboard_dir)
    summary_writer = tf.summary.create_file_writer(tensorboard_dir)
  else:
    summary_writer = None

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

  backend.set_image_data_format(image_format)
  dtype='float16' if precision == 'fp16' else 'float32'
  backend.set_floatx(dtype)
  model = model_func(num_classes=image_processing.NUM_CLASSES,
                     batch_size=batch_size)

  loss_func = keras.losses.SparseCategoricalCrossentropy()

  train_top1 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1,
                                                              name='train_top1')
  train_top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5,
                                                              name='train_top5')

  val_loss = tf.keras.metrics.Mean(name='val_loss', dtype=tf.float32)

  val_top1 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1,
                                                            name='val_top1')
  val_top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5,
                                                            name='val_top5')

  if log_dir:
    # We save check points only when using the real data.
    assert data_dir, "--data_dir cannot be empty when using --log_dir"
    assert os.path.exists(log_dir)
    ckpt = tf.train.Checkpoint(epoch=tf.Variable(0), optimizer=opt, net=model)
    manager = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=3,
                                         checkpoint_name="model-ckpt")

  @tf.function
  def train_step(inputs, first_batch):
    images, labels = inputs

    with tf.GradientTape() as tape:
      predictions = model(images, training=True)
      loss = loss_func(labels, predictions)
      loss += tf.reduce_sum(model.losses)
      loss_copy = loss
      # Scale the losses
      if precision == 'fp16':
        loss = loss * tf.cast(loss_scale, loss.dtype)

    tape = hvd.DistributedGradientTape(tape)

    old_grads = tape.gradient(loss, model.trainable_variables)

    # Unscale the grads
    if precision == 'fp16':
      loss_scale_reciprocal = 1. / loss_scale
      grads = [g * tf.cast(loss_scale_reciprocal, g.dtype) if g is not
                           None else None for g in old_grads]
    else:
      grads = old_grads

    opt.apply_gradients(zip(grads, model.trainable_variables))

    train_top1.update_state(labels, predictions)
    train_top5.update_state(labels, predictions)

    if hvd.size() > 1 and first_batch:
      hvd.broadcast_variables(model.variables, root_rank=0)
      hvd.broadcast_variables(opt.variables(), root_rank=0)

    return loss_copy

  @tf.function
  def valid_step(inputs):
    images, labels = inputs
    predictions = model(images, training=False)
    loss = loss_func(labels, predictions)

    val_loss.update_state(loss)
    val_top1.update_state(labels, predictions)
    val_top5.update_state(labels, predictions)

  if data_dir is not None:
    num_preproc_threads = 4 if dali_mode else 10
    train_input = image_processing.image_set(train_files, batch_size,
        image_height, image_width, training=True, distort_color=distort_color,
        deterministic=False, num_threads=num_preproc_threads,
        use_dali=dali_mode, idx_filenames=train_idx_files)

    valid_input = image_processing.image_set(valid_files, batch_size,
        image_height, image_width, training=False, distort_color=False,
        deterministic=False, num_threads=num_preproc_threads,
        use_dali=dali_mode, idx_filenames=valid_idx_files)
  else:
    if dali_mode:
      raise ValueError("Must provide --data_dir if Dali is enabled")
    else:
      train_input = image_processing.fake_image_set(batch_size, image_height,
                                                    image_width)

  global_steps = 0
  log_steps = display_every
  try:

    initial_epoch = 0
    if log_dir:
      ckpt.restore(manager.latest_checkpoint)
      if manager.latest_checkpoint:
        if hvd.rank() == 0:
          print("Restored from {}".format(manager.latest_checkpoint))
        initial_epoch = max(
            int(re.findall(r'\d+', manager.latest_checkpoint)[0]),
                initial_epoch)
      else:
        if hvd.rank() == 0:
          print("Initializing from scratch.")

    # Training Loop
    for epoch in range(num_epochs):
      if epoch < initial_epoch:
        continue
      # on_epoch_begin
      epoch_start = time.time()

      total_loss = 0.0
      num_batches = 0
      train_top1.reset_states()
      train_top5.reset_states()

      if not dali_mode:
        train_iter = iter(train_input)
      for _ in range(nstep_per_epoch):
        # on_batch_begin
        global_steps += 1
        if global_steps == 1:
          start_time = time.time()

        if global_steps == 1 and hvd.rank() == 0 and summary_writer:
          tf.summary.trace_on(graph=True, profiler=True)

        if not dali_mode:
          x = next(train_iter)
        else:
          x = train_input.get_device_minibatches()
        total_loss += train_step(x, global_steps == 1)

        if global_steps == 1 and hvd.rank() == 0 and summary_writer:
          with summary_writer.as_default():
            tf.summary.trace_export(name="train_step", step=0,
                                    profiler_outdir=tensorboard_dir)

        # on_batch_end
        if global_steps % log_steps == 0:
          timestamp = time.time()
          elapsed_time = timestamp - start_time
          examples_per_second = \
              (batch_size * hvd.size() * log_steps) / elapsed_time
          if hvd.rank() == 0:
            print("global_step: %d images_per_sec: %.1f" % (global_steps,
                  examples_per_second))
          start_time = timestamp
        num_batches += 1

      train_loss = total_loss / num_batches

      # on_epoch_end
      epoch_run_time = time.time() - epoch_start
      if hvd.rank() == 0:
        print("epoch: %d time_taken: %.1f" % (epoch, epoch_run_time))

      if data_dir is not None:
        val_loss.reset_states()
        val_top1.reset_states()
        val_top5.reset_states()

        if not dali_mode:
          test_iter = iter(valid_input)
        for _ in range(nstep_per_valid):
          if not dali_mode:
            x = next(test_iter)
          else:
            x = valid_input.get_device_minibatches()
          valid_step(x)

      if log_dir:
        ckpt.epoch.assign_add(1)
        if hvd.rank() == 0:
          save_path = manager.save()
          print("Saved checkpoint for epoch {}: {}".format(int(ckpt.epoch),
                                                           save_path))

      if hvd.rank() == 0:
        output_str = ("loss: {} - top1: {} - top5: {} - val_loss: {} - "
                      "val_top1: {} - val_top5: {}")
        print(output_str.format(train_loss, train_top1.result(),
                                train_top5.result(), val_loss.result(),
                                val_top1.result(), val_top5.result()))

      if hvd.rank() == 0 and summary_writer:
        with summary_writer.as_default():
          tf.summary.scalar('train_loss', train_loss, global_steps)
          tf.summary.scalar('train_top1', train_top1.result(), global_steps)
          tf.summary.scalar('train_top5', train_top5.result(), global_steps)
          tf.summary.scalar('val_loss', val_loss.result(), global_steps)
          tf.summary.scalar('val_top1', val_top1.result(), global_steps)
          tf.summary.scalar('val_top5', val_top5.result(), global_steps)

    if hvd.rank() == 0 and summary_writer:
      summary_writer.close()

  except KeyboardInterrupt:
    print("Keyboard interrupt")

  if export_dir and hvd.rank() == 0:
    model.save(save_format)
    print(f"The model is saved to {save_format}")

def predict_ctl(params):
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

