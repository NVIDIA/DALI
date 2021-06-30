import multiprocessing
from absl import logging
import numpy as np
import tensorflow as tf
import random
import re
import os

import hparams_config
import utils
from model import efficientdet_net
from model.utils import optimizers


def run_training(args):
    logging.set_verbosity(logging.WARNING)

    args = utils.dict_to_namedtuple(args)

    config = hparams_config.get_efficientdet_config(args.model_name)
    config.override(args.hparams, allow_new_keys=True)
    config.image_size = utils.parse_image_size(config.image_size)

    params = dict(
        config.as_dict(),
        seed=args.seed,
        train_batch_size=args.train_batch_size,
    )

    logging.info(params)

    if args.ckpt_dir:
        ckpt_dir = args.ckpt_dir
        if not tf.io.gfile.exists(ckpt_dir):
            tf.io.gfile.makedirs(ckpt_dir)
        config_file = os.path.join(ckpt_dir, "config.yaml")
        if not tf.io.gfile.exists(config_file):
            tf.io.gfile.GFile(config_file, "w").write(str(config))

    if params["seed"]:
        seed = params["seed"]
        os.environ["PYTHONHASHSEED"] = str(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    physical_devices = tf.config.list_physical_devices("GPU")
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)
    tf.config.set_soft_device_placement(True)

    num_devices = 1
    multi_gpu = args.multi_gpu
    if multi_gpu is not None and len(multi_gpu) != 1 and len(physical_devices) > 1:
        devices = [f"GPU:{gpu}" for gpu in multi_gpu] if len(multi_gpu) != 0 else None
        strategy = tf.distribute.MirroredStrategy(devices)
        num_devices = len(devices) if devices else len(physical_devices)
    else:
        strategy = tf.distribute.get_strategy()

    train_dataset = utils.get_dataset(
        args,
        args.train_batch_size * num_devices,
        True,
        params,
        strategy if num_devices > 1 else None,
    )

    if args.eval_after_training or args.eval_during_training:
        eval_dataset = utils.get_dataset(
            args,
            num_devices,
            False,
            params,
            strategy if num_devices > 1 else None,
        )
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )
        eval_dataset = eval_dataset.with_options(options)

    with strategy.scope():
        model = efficientdet_net.EfficientDetNet(params=params)

        global_batch_size = args.train_batch_size * strategy.num_replicas_in_sync
        model.compile(
            optimizer=optimizers.get_optimizer(
                params, args.epochs, global_batch_size, args.train_steps
            )
        )

        initial_epoch = args.initial_epoch
        if args.start_weights:
            image_size = params["image_size"]
            model.predict(np.zeros((1, image_size[0], image_size[1], 3)))
            model.load_weights(args.start_weights)
            fname = args.start_weights.split("/")[-1]
            ckpt_pattern = f"{args.model_name}\.(\d\d+)\.h5"
            match = re.match(ckpt_pattern, fname)
            if match:
                initial_epoch = int(match.group(1).lstrip("0"))

        callbacks = []

        if args.ckpt_dir:
            ckpt_dir = args.ckpt_dir
            if not tf.io.gfile.exists(ckpt_dir):
                tf.io.gfile.makedirs(tensorboard_dir)
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join(
                        ckpt_dir, "".join([args.model_name, ".{epoch:02d}.h5"])
                    ),
                    save_weights_only=True,
                )
            )

        if args.log_dir:
            log_dir = args.log_dir
            if not tf.io.gfile.exists(log_dir):
                tf.io.gfile.makedirs(log_dir)
            callbacks.append(
                tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq="epoch")
            )

        model.fit(
            train_dataset,
            epochs=args.epochs,
            steps_per_epoch=args.train_steps,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            validation_data=eval_dataset if args.eval_during_training else None,
            validation_steps=args.eval_steps,
            validation_freq=args.eval_freq,
        )

        if args.eval_after_training:
            print("Evaluation after training:")
            model.evaluate(eval_dataset, steps=args.eval_steps)

        model.save_weights(args.output)


if __name__ == "__main__":
    import argparse
    from argparse_utils import enum_action

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial_epoch",
        type=int,
        default=0,
        help="epoch from which to start training",
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="epoch on which training should finish"
    )
    parser.add_argument(
        "--input",
        action=enum_action(utils.InputType),
        required=True,
        help="input type",
    )
    parser.add_argument(
        "--images_path",
        required=False,
        help="Path to COCO images",
    )
    parser.add_argument(
        "--annotations_path",
        required=False,
        help="Path to COCO annotations",
    )
    parser.add_argument(
        "--train_file_pattern",
        required=False,
        help="glob pattern for TFrecord files with training data",
    )
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument(
        "--train_steps", type=int, default=2000, help="number of steps in each epoch"
    )
    parser.add_argument(
        "--eval_file_pattern",
        help="glob pattern for TFrecord files with evaluation data, "
        "defaults to TRAIN_FILE_PATTERN if not given",
    )
    parser.add_argument(
        "--eval_steps", type=int, default=5000, help="number of examples to evaluate"
    )
    parser.add_argument(
        "--eval_freq", type=int, default=1, help="during training evalutaion frequency"
    )
    parser.add_argument(
        "--eval_during_training",
        action="store_true",
        help="whether to run evaluation every EVAL_FREQ epochs",
    )
    parser.add_argument(
        "--eval_after_training",
        action="store_true",
        help="whether to run evaluation after finished training",
    )
    parser.add_argument(
        "--pipeline",
        action=enum_action(utils.PipelineType),
        required=True,
        help="pipeline type",
    )
    parser.add_argument(
        "--multi_gpu",
        nargs="*",
        type=int,
        help="list of GPUs to use, defaults to all visible GPUs",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--hparams", default="", help="string or filename with parameters"
    )
    parser.add_argument("--model_name", default="efficientdet-d1")
    parser.add_argument(
        "--output", default="output.h5", help="filename for final weights to save"
    )
    parser.add_argument("--start_weights")
    parser.add_argument("--log_dir", help="directory for tensorboard logs")
    parser.add_argument("--ckpt_dir", help="directory for saving weights each step")

    args = parser.parse_args()
    run_training(vars(args))
