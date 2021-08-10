import multiprocessing
from absl import logging
import numpy as np
import tensorflow as tf
import re
import os

import hparams_config
import utils
from model import efficientdet_net


def run_eval(args):
    logging.set_verbosity(logging.WARNING)

    args = utils.dict_to_namedtuple(args)

    config = hparams_config.get_efficientdet_config(args.model_name)
    config.override(args.hparams, allow_new_keys=True)
    config.image_size = utils.parse_image_size(config.image_size)

    params = dict(config.as_dict(), seed=None)

    logging.info(params)

    utils.setup_gpus()

    dataset = utils.get_dataset(args, 1, False, params, None)

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

    parser = argparse.ArgumentParser(formatter_class=utils.SmartFormatter)
    parser.add_argument(
        "--input_type",
        action=enum_action(utils.InputType),
        required=True,
        help="Input type.",
    )
    parser.add_argument(
        "--images_path",
        help="Path to COCO images.",
    )
    parser.add_argument(
        "--annotations_path",
        help="Path to COCO annotations.",
    )
    parser.add_argument(
        "--eval_file_pattern",
        help="TFrecord files glob pattern for files with evaluation data.",
    )
    parser.add_argument(
        "--eval_steps", type=int, default=5000, help="Number of examples to evaluate."
    )
    parser.add_argument(
        "--pipeline_type",
        action=enum_action(utils.PipelineType),
        required=True,
        help="R|Pipeline type used while loading and preprocessing data. One of:\n"
        "tensorflow – pipeline used in original EfficientDet implementation on https://github.com/google/automl/tree/master/efficientdet;\n"
        "synthetic – like `tensorflow` pipeline type but repeats one batch endlessly;\n"
        "dali_gpu – pipeline which uses Nvidia Data Loading Library (DALI) to run part of data preprocessing on GPUs to improve efficiency;\n"
        "dali_cpu – like `dali_gpu` pipeline type but restricted to run only on CPU.",
    )
    parser.add_argument(
        "--weights", default="output.h5", help="Name of the file with model weights."
    )
    parser.add_argument("--model_name", default="efficientdet-d1")
    parser.add_argument(
        "--hparams", default="", help="String or filename with parameters."
    )

    args = parser.parse_args()
    run_eval(vars(args))
