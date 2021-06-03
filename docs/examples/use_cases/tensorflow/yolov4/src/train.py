from model import YOLOv4Model, calc_loss
import numpy as np
import tensorflow as tf

from img import read_img, draw_img
from dali.pipeline import YOLOv4Pipeline
from np.pipeline import YOLOv4PipelineNumpy
import utils

import math
import os
import random


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


def train(file_root, annotations_file, batch_size, epochs, steps_per_epoch, **kwargs):

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

    total_steps = epochs * steps_per_epoch
    initial_lr = kwargs.get("lr")
    lr_fn = YOLOLearningRateSchedule(initial_lr)

    initial_epoch = 0

    multigpu = kwargs.get("multigpu")
    strategy = tf.distribute.MirroredStrategy() if multigpu else tf.distribute.get_strategy()

    with strategy.scope():
        model = YOLOv4Model()
        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=lr_fn)
        )

    if start_weights:
        model.load_weights(start_weights)
        fn = start_weights.split('/')[-1]
        if fn.endswith('.h5') and fn.startswith('epoch_'):
            initial_epoch = int(fn[6 : -3])


    def dataset_fn(input_context):
        image_size = (608, 608)
        device_id = input_context.input_pipeline_id
        num_threads = input_context.num_input_pipelines

        if pipeline == 'dali-gpu' or pipeline == 'dali-cpu':
            with tf.device("/gpu:{}".format(input_context.input_pipeline_id)):
                yolo = YOLOv4Pipeline(
                    file_root, annotations_file,
                    batch_size, image_size, num_threads, device_id, seed,
                    use_gpu=pipeline == 'dali-gpu',
                    is_training=True,
                    use_mosaic=use_mosaic
                )
                return yolo.dataset()

        if pipeline == 'numpy':
            yolo = YOLOv4PipelineNumpy(
                file_root, annotations_file,
                batch_size, image_size, num_threads, device_id, seed,
                is_training=True,
                use_mosaic=use_mosaic
            )
            return yolo.dataset()

    input_options = tf.distribute.InputOptions(
        experimental_place_dataset_on_device = True,
        experimental_prefetch_to_device = False,
        experimental_replication_mode = tf.distribute.InputReplicationMode.PER_REPLICA)

    dataset = strategy.distribute_datasets_from_function(dataset_fn, input_options)


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
        callbacks=callbacks
    )

    return model
