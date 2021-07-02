# Copyright 2020 Google Research. All Rights Reserved.
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
"""The main training script."""
import multiprocessing
import os
from absl import app
from absl import flags
from absl import logging
import numpy as np

logging.set_verbosity(logging.WARNING)
import tensorflow as tf

from pipeline.tf import dataloader
import hparams_config
import utils
from model import efficientdet_train

flags.DEFINE_enum(
    "strategy", None, ["gpus"], "Training: gpus for multi-gpu, if None, use TF default."
)
flags.DEFINE_enum(
    "mode",
    "train",
    ["train", "eval", "train_and_eval"],
    "Mode to run: train, eval or train_and_eval (default: train)",
)
flags.DEFINE_bool("use_fake_data", False, "Use fake input.")
flags.DEFINE_string("model_dir", None, "Location of model_dir")
flags.DEFINE_string("tensorboard_dir", None, "Location of tensorboard log_dir")
flags.DEFINE_string(
    "backbone_ckpt",
    "",
    "Location of the ResNet50 checkpoint to use for model " "initialization.",
)
flags.DEFINE_string("ckpt", None, "Start training from this EfficientDet checkpoint.")
flags.DEFINE_string(
    "hparams",
    "",
    "Comma separated k=v pairs of hyperparameters or a module"
    " containing attributes to use as hyperparameters.",
)
flags.DEFINE_integer("num_cores", default=1, help="Number of GPU cores for training")
flags.DEFINE_integer("train_batch_size", 64, "global training batch size")
flags.DEFINE_integer("eval_batch_size", 1, "global evaluation batch size")
flags.DEFINE_integer("eval_samples", 5000, "Number of samples for eval.")
flags.DEFINE_integer(
    "save_checkpoints_steps", 100, "Number of iterations per checkpoint save"
)
flags.DEFINE_string(
    "training_file_pattern",
    None,
    "Glob for training data files (e.g., COCO train - minival set)",
)
flags.DEFINE_string(
    "validation_file_pattern",
    None,
    "Glob for evaluation tfrecords (e.g., COCO val2017 set)",
)
flags.DEFINE_string(
    "val_json_file",
    None,
    "COCO validation JSON containing golden bounding boxes. If None, use the "
    "ground truth from the dataloader. Ignored if testdev_dir is not None.",
)
flags.DEFINE_string(
    "testdev_dir", None, "COCO testdev dir. If not None, ignore val_json_file."
)
flags.DEFINE_integer(
    "num_examples_per_epoch", 120000, "Number of examples in one epoch"
)
flags.DEFINE_integer("num_epochs", None, "Number of epochs for training")
flags.DEFINE_string("model_name", "efficientdet-d1", "Model name.")
flags.DEFINE_bool(
    "eval_after_training", False, "Run one eval after the " "training finishes."
)
flags.DEFINE_bool("profile", False, "Profile training performance.")
flags.DEFINE_integer(
    "tf_random_seed",
    None,
    "Sets the TF graph seed for deterministic execution"
    " across runs (for debugging).",
)

# For Eval mode
flags.DEFINE_integer("min_eval_interval", 180, "Minimum seconds between evaluations.")
flags.DEFINE_integer(
    "eval_timeout",
    None,
    "Maximum seconds between checkpoints before evaluation terminates.",
)

# for train_and_eval mode
flags.DEFINE_bool(
    "run_epoch_in_child_process",
    False,
    "This option helps to rectify CPU memory leak. If True, every epoch is "
    "run in a separate process for train and eval and memory will be cleared."
    "Drawback: need to kill 2 processes if trainining needs to be interrupted.",
)

FLAGS = flags.FLAGS


def main(_):
    # Check data path
    if FLAGS.mode in ("train", "train_and_eval"):
        if FLAGS.training_file_pattern is None:
            raise RuntimeError("Must specify --training_file_pattern for train.")
    if FLAGS.mode in ("eval", "train_and_eval"):
        if FLAGS.validation_file_pattern is None:
            raise RuntimeError("Must specify --validation_file_pattern for eval.")

    # Parse and override hparams
    config = hparams_config.get_detection_config(FLAGS.model_name)
    config.override(FLAGS.hparams)
    if FLAGS.num_epochs:  # NOTE: remove this flag after updating all docs.
        config.num_epochs = FLAGS.num_epochs

    # Parse image size in case it is in string format.
    config.image_size = utils.parse_image_size(config.image_size)

    model_dir = FLAGS.model_dir
    if not tf.io.gfile.exists(model_dir):
        tf.io.gfile.makedirs(model_dir)

    config_file = os.path.join(model_dir, "config.yaml")
    if not tf.io.gfile.exists(config_file):
        tf.io.gfile.GFile(config_file, "w").write(str(config))

    params = dict(
        config.as_dict(),
        model_name=FLAGS.model_name,
        model_dir=FLAGS.model_dir,
        num_shards=FLAGS.num_cores,
        num_examples_per_epoch=FLAGS.num_examples_per_epoch,
        strategy=FLAGS.strategy,
        backbone_ckpt=FLAGS.backbone_ckpt,
        ckpt=FLAGS.ckpt,
        val_json_file=FLAGS.val_json_file,
        testdev_dir=FLAGS.testdev_dir,
        profile=FLAGS.profile,
        mode=FLAGS.mode,
    )
    logging.info(params)

    if FLAGS.eval_samples:
        eval_steps = int(
            (FLAGS.eval_samples + FLAGS.eval_batch_size - 1) // FLAGS.eval_batch_size
        )
    else:
        eval_steps = None

    max_instances_per_image = params["max_instances_per_image"]
    train_input_fn = dataloader.InputReader(
        FLAGS.training_file_pattern,
        is_training=True,
        use_fake_data=FLAGS.use_fake_data,
        max_instances_per_image=max_instances_per_image,
    )
    eval_input_fn = dataloader.InputReader(
        FLAGS.validation_file_pattern,
        is_training=False,
        use_fake_data=FLAGS.use_fake_data,
        max_instances_per_image=max_instances_per_image,
    )

    if FLAGS.strategy == "gpus":
        strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2"])

    else:
        strategy = None


    params["batch_size"] = FLAGS.train_batch_size // params["num_shards"]

    model = efficientdet_train.EfficientDetTrain(params=params)
    model.compile(optimizer=efficientdet_train.get_optimizer(params))

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "model.{epoch:02d}.hdf5"),
        save_freq="epoch",  # FLAGS.save_checkpoints_steps,
        options=tf.saved_model.SaveOptions(experimental_io_device=True),
    )

    callbacks = [model_checkpoint_callback]

    if FLAGS.tensorboard_dir:
        tensorboard_dir = FLAGS.tensorboard_dir
        if not tf.io.gfile.exists(tensorboard_dir):
            tf.io.gfile.makedirs(tensorboard_dir)
        callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, update_freq="epoch")
        )

    train_steps = params["num_examples_per_epoch"] // params["batch_size"]

    # start train/eval flow.
    if FLAGS.mode == "train":
        model.fit(
            train_input_fn(params),
            epochs=params["num_epochs"],
            batch_size=params["batch_size"],
            steps_per_epoch=train_steps,
            callbacks=callbacks,
        )
    elif FLAGS.mode == "eval":
        raise ValueError("eval mode is not available yet :<")
    elif FLAGS.mode == "train_and_eval":
        raise ValueError("train_and_eval mode is not available yet :<")
    else:
        logging.info("Invalid mode: %s", FLAGS.mode)


if __name__ == "__main__":
    app.run(main)
