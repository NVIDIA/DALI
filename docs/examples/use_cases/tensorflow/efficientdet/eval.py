import multiprocessing
from absl import logging
import numpy as np
import tensorflow as tf
import re
import os

import hparams_config
import utils
from model import efficientdet_net
from pipeline.dali import fn_pipeline


def run_eval(args):
    logging.set_verbosity(logging.WARNING)

    args = utils.dict_to_namedtuple(args)

    physical_devices = tf.config.list_physical_devices("GPU")
    for gpu_instance in physical_devices:
        tf.config.experimental.set_memory_growth(gpu_instance, True)
    tf.config.set_soft_device_placement(True)

    config = hparams_config.get_efficientdet_config(args.model_name)
    config.override(args.hparams, allow_new_keys=True)
    config.image_size = utils.parse_image_size(config.image_size)

    params = dict(config.as_dict(), seed=None)

    logging.info(params)

    dataset = utils.get_dataset(
        args.pipeline, args.eval_file_pattern, 1, False, params, None
    )

    model = efficientdet_net.EfficientDetNet(params=params)
    model.compile()

    if args.weights:
        image_size = params["image_size"]
        model.predict(np.zeros((1, image_size[0], image_size[1], 3)))
        model.load_weights(args.weights)

    model.evaluate(dataset, steps=args.eval_steps)


if __name__ == "__main__":
    import argparse
    from argparse_utils import enum_action

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_file_pattern",
        required=True,
        help="glob pattern for TFrecord files with evaluation data",
    )
    parser.add_argument(
        "--eval_steps", type=int, default=5000, help="number of examples to evaluate"
    )
    parser.add_argument(
        "--pipeline",
        action=enum_action(utils.PipelineType),
        required=True,
        help="pipeline type",
    )
    parser.add_argument(
        "--weights", default="output.h5", help="file with model weights"
    )
    parser.add_argument("--model_name", default="efficientdet-d1")
    parser.add_argument(
        "--hparams", default="", help="string or filename with parameters"
    )

    args = parser.parse_args()
    run_eval(vars(args))
