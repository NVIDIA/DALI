from model import YOLOv4Model, calc_loss
import numpy as np
import tensorflow as tf

from img import read_img, draw_img
from pipeline import YOLOv4Pipeline
import utils

import math
import os


SET_MEMORY_GROWTH = False


class SaveWeightsCallback(tf.keras.callbacks.Callback):

    def __init__(self, ckpt_dir):
        self.ckpt_dir = ckpt_dir

    def on_epoch_begin(self, epoch, logs=None):
        self.model.save_weights(self.ckpt_dir + '/epoch_' + str(epoch) + '.h5')


# TODO: add multigpu strategy
# TODO: fix nan loss issue
def train(file_root, annotations_file, batch_size, epochs, steps_per_epoch, **kwargs):

    if SET_MEMORY_GROWTH:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    use_gpu = kwargs.get("use_gpu", True)
    log_dir = kwargs.get("log_dir")
    ckpt_dir = kwargs.get("ckpt_dir")
    start_weights = kwargs.get("start_weights")
    initial_epoch = 0

    model = YOLOv4Model()
    if start_weights:
        model.load_weights(start_weights)
        fn = start_weights.split('/')[-1]
        if fn.endswith('.h5') and fn.startswith('epoch_'):
            initial_epoch = int(fn[6 : -3])

    image_size = (608, 608)
    num_threads = 1
    device_id = 0
    seed = int.from_bytes(os.urandom(4), "little")

    pipeline = YOLOv4Pipeline(
        file_root, annotations_file, batch_size, image_size, num_threads, device_id, seed, use_gpu, True
    )
    dataset = pipeline.dataset()

    callbacks = []
    if log_dir:
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            update_freq='epoch'
        ))
    if ckpt_dir:
        callbacks.append(SaveWeightsCallback(ckpt_dir))

    model.compile(
        optimizer=tf.keras.optimizers.Adam()
    )

    model.fit(
        pipeline.dataset(),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        initial_epoch=initial_epoch,
        callbacks=callbacks
    )

    return model
