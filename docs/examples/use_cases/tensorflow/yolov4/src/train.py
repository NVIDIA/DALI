# Copyright 2021 Kacper Kluk. All Rights Reserved.
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

from model import YOLOv4Model
import numpy as np
import tensorflow as tf

from dali.pipeline import YOLOv4Pipeline
from np.pipeline import YOLOv4PipelineNumpy

import os
import random
import atexit


SET_MEMORY_GROWTH = True


class SaveWeightsCallback(tf.keras.callbacks.Callback):

    def __init__(self, ckpt_dir):
        self.ckpt_dir = ckpt_dir

    def on_epoch_begin(self, epoch, logs=None):
        self.model.save_weights(self.ckpt_dir + '/epoch_' + str(epoch) + '.h5')

class YOLOLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, init_lr):
        self.init_lr = init_lr

    def __call__(self, step):
        warmup = tf.math.minimum(1.0, tf.cast(step, tf.float32) / 1000)
        return warmup * self.init_lr



def train(file_root, annotations, batch_size, epochs, steps_per_epoch, **kwargs):

    seed = kwargs.get("seed")
    if not seed:
        seed = int.from_bytes(os.urandom(4), "little")
    else:
        os.environ['PYTHONHASHSEED']=str(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


    if SET_MEMORY_GROWTH:
        pds = tf.config.list_physical_devices('GPU')
        for pd in pds:
            tf.config.experimental.set_memory_growth(pd, True)

    pipeline = kwargs.get("pipeline")
    use_mosaic = kwargs.get("use_mosaic")
    log_dir = kwargs.get("log_dir")
    ckpt_dir = kwargs.get("ckpt_dir")
    start_weights = kwargs.get("start_weights")

    def get_dataset_fn(file_root, annotations,
                       batch_size, pipeline, is_training):

        def dataset_fn(input_context):
            image_size = (608, 608)
            device_id = input_context.input_pipeline_id
            num_threads = input_context.num_input_pipelines

            if pipeline == 'dali-gpu' or pipeline == 'dali-cpu':
                with tf.device("/gpu:{}".format(input_context.input_pipeline_id)):
                    yolo = YOLOv4Pipeline(
                        file_root, annotations,
                        batch_size, image_size, num_threads, device_id, seed,
                        use_gpu=pipeline == 'dali-gpu',
                        is_training=is_training,
                        use_mosaic=use_mosaic
                    )
                    return yolo.dataset()

            if pipeline == 'numpy':
                yolo = YOLOv4PipelineNumpy(
                    file_root, annotations,
                    batch_size, image_size, num_threads, device_id, seed,
                    is_training=is_training,
                    use_mosaic=use_mosaic
                )
                return yolo.dataset()

        return dataset_fn


    total_steps = epochs * steps_per_epoch
    initial_lr = kwargs.get("lr")
    lr_fn = YOLOLearningRateSchedule(initial_lr)

    initial_epoch = 0

    multigpu = kwargs.get("multigpu")
    strategy = tf.distribute.MirroredStrategy() if multigpu else tf.distribute.get_strategy()
    if hasattr(strategy._extended._collective_ops, "_pool"):
        atexit.register(strategy._extended._collective_ops._pool.close) # type: ignore

    with strategy.scope():
        model = YOLOv4Model()
        model.compile(
            optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=lr_fn)
        )

    if start_weights:
        model.load_weights(start_weights)
        fn = start_weights.split('/')[-1]
        if fn.endswith('.h5') and fn.startswith('epoch_'):
            initial_epoch = int(fn[6 : -3])

    input_options = tf.distribute.InputOptions(
        experimental_place_dataset_on_device = True,
        experimental_fetch_to_device = False,
        experimental_replication_mode = tf.distribute.InputReplicationMode.PER_REPLICA)

    dataset = strategy.distribute_datasets_from_function(
        get_dataset_fn(file_root, annotations, batch_size, pipeline, True),
        input_options)

    eval_file_root = kwargs.get('eval_file_root')
    eval_annotations = kwargs.get('eval_annotations')
    eval_dataset = None
    if not eval_file_root is None and not eval_annotations is None:
        eval_dataset = strategy.distribute_datasets_from_function(
            get_dataset_fn(eval_file_root, eval_annotations, 1, 'dali-cpu', False),
            tf.distribute.InputOptions()
        )


    callbacks = []
    if log_dir:
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            update_freq='epoch'
        ))
    if ckpt_dir:
        callbacks.append(SaveWeightsCallback(ckpt_dir))


    model.fit(
        dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        validation_data=eval_dataset,
        validation_steps=kwargs.get('eval_steps'),
        validation_freq=kwargs.get('eval_frequency'),
    )

    return model
