import multiprocessing
from absl import logging
import numpy as np
import tensorflow as tf
import re
import os

from enum import Enum

import hparams_config
import utils
from model import efficientdet_train


class PipelineType(Enum):
    syntetic = 0
    tensorflow = 1
    dali_cpu = 2
    dali_gpu = 3


def run_training(args):
    logging.set_verbosity(logging.WARNING)

    args = utils.dict_to_namedtuple(args)

    eval_file_pattern = args.eval_file_pattern
    eval_in_fit = True if eval_file_pattern else False
    if args.eval_after_training and not eval_file_pattern:
        eval_file_pattern = args.train_file_pattern

    config = hparams_config.get_efficientdet_config(args.model_name)
    config.override(args.hparams, allow_new_keys=True)
    config.image_size = utils.parse_image_size(config.image_size)

    params = dict(
        config.as_dict(),
        seed=args.seed,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
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

    use_mirrored_strategy = False
    multi_gpu = args.multi_gpu
    if multi_gpu is not None and len(physical_devices) > 1:
        devices = [f"GPU:{gpu}" for gpu in multi_gpu] if multi_gpu else None
        strategy = tf.distribute.MirroredStrategy(devices)
        use_mirrored_strategy = True
    else:
        strategy = tf.distribute.get_strategy()

    train_dataset = None
    eval_dataset = None
    pipeline = args.pipeline

    if pipeline in [PipelineType.tensorflow, PipelineType.syntetic]:
        from pipeline.tf.dataloader import InputReader

        train_dataset = InputReader(
            params,
            args.train_file_pattern,
            is_training=True,
            use_fake_data=(pipeline == PipelineType.syntetic),
        ).get_dataset(args.train_batch_size)

        if eval_file_pattern:
            eval_dataset = InputReader(
                params,
                eval_file_pattern,
            ).get_dataset(args.eval_batch_size)

    elif use_mirrored_strategy:
        from pipeline.dali.fn_pipeline import EfficientDetPipeline
        from functools import partial

        if pipeline == PipelineType.dali_cpu:
            raise ValueError(
                "dali_cpu pipeline is not compatible with mulit_gpu mode :<"
            )

        def dali_dataset_fn(batch_size, file_pattern, is_training, input_context):
            with tf.device(f"/gpu:{input_context.input_pipeline_id}"):
                device_id = input_context.input_pipeline_id
                num_shards = input_context.num_input_pipelines
                return EfficientDetPipeline(
                    params,
                    batch_size,
                    file_pattern,
                    is_training=is_training,
                    num_shards=num_shards,
                    device_id=device_id,
                ).get_dataset()

        input_options = tf.distribute.InputOptions(
            experimental_place_dataset_on_device=True,
            experimental_prefetch_to_device=False,
            experimental_replication_mode=tf.distribute.InputReplicationMode.PER_REPLICA,
        )

        train_dataset = strategy.distribute_datasets_from_function(
            partial(
                dali_dataset_fn, args.train_batch_size, args.train_file_pattern, True
            ),
            input_options,
        )

        if eval_file_pattern:
            eval_dataset = strategy.distribute_datasets_from_function(
                partial(
                    dali_dataset_fn, args.eval_batch_size, eval_file_pattern, False
                ),
                input_options,
            )

    else:
        from pipeline.dali.fn_pipeline import EfficientDetPipeline

        cpu_only = pipeline == PipelineType.dali_cpu
        device = "/cpu:0" if cpu_only else "/gpu:0"
        with tf.device(device):
            train_dataset = EfficientDetPipeline(
                params,
                args.train_batch_size,
                args.train_file_pattern,
                is_training=True,
                cpu_only=cpu_only,
            ).get_dataset()

            if eval_file_pattern:
                eval_dataset = EfficientDetPipeline(
                    params,
                    args.eval_batch_size,
                    eval_file_pattern,
                    is_training=False,
                    cpu_only=cpu_only,
                ).get_dataset()

    with strategy.scope():
        model = efficientdet_train.EfficientDetTrain(params=params)

        global_batch_size = args.train_batch_size * strategy.num_replicas_in_sync
        model.compile(
            optimizer=efficientdet_train.get_optimizer(
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

        # if params["moving_average_decay"]:
        #     from tensorflow_addons import (
        #         callbacks as tfa_callbacks,
        #     )  # pylint: disable=g-import-not-at-top

        #     callbacks.append(
        #         tfa_callbacks.AverageModelCheckpoint(
        #             filepath=os.path.join(
        #                 ckpt_dir, "".join([args.model_name, ".avg.{epoch:02d}.h5"])
        #             ),
        #             save_weights_only=True,
        #             update_weights=False,
        #         )
        #     )

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
            validation_data=eval_dataset if eval_in_fit else None,
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
    parser.add_argument("--initial_epoch", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--train_file_pattern", required=True)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--train_steps", type=int, default=100)
    parser.add_argument("--eval_file_pattern")
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_steps", type=int)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--eval_after_training", action="store_true")
    parser.add_argument("--pipeline", action=enum_action(PipelineType), required=True)
    parser.add_argument("--multi_gpu", nargs="*", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--hparams", default="")
    parser.add_argument("--model_name", default="efficientdet-d1")
    parser.add_argument("--output", default="output.h5")
    parser.add_argument("--start_weights")
    parser.add_argument("--log_dir")
    parser.add_argument("--ckpt_dir")

    args = parser.parse_args()

    print(args)

    run_training(vars(args))
