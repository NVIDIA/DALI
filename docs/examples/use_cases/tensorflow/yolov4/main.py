from model import YOLOv4Model
from pipeline import YOLOv4Pipeline
import numpy as np

from img import read_img, draw_img, save_img, add_bboxes
import inference
import train

import math
import sys
import os


def run_infer(weights_file, labels_file, image_path, out_filename):

    cls_names = open(labels_file, "r").read().split("\n")

    model = YOLOv4Model()
    model.load_weights(weights_file)

    img, input = read_img(image_path, 608)

    prediction = model.predict(input)
    boxes, scores, labels = inference.decode_prediction(prediction, len(cls_names))

    labels = [cls_names[cls] for cls in labels]

    pixels = add_bboxes(img, boxes, scores, labels)
    if out_filename:
        save_img(out_filename, pixels)
    else:
        draw_img(pixels)


def run_training(file_root, annotations, batch_size, epochs, steps_per_epoch, **kwargs):

    model = train.train(file_root, annotations, batch_size, epochs, steps_per_epoch, **kwargs)

    output = kwargs.get("output")
    if output:
        model.save_weights(output)


def run_eval(file_root, annotations_file, weights_file, batch_size, steps):

    model = YOLOv4Model()
    model.load_weights(weights_file)

    seed = int.from_bytes(os.urandom(4), "little")

    pipeline = YOLOv4Pipeline(
        file_root, annotations_file, batch_size, (608, 608), 1, 0, seed, True, False
    )
    dataset = pipeline.dataset()

    model.compile(run_eagerly=True)
    model.evaluate(pipeline.dataset(), steps=steps)



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action")
    subparsers.required = True

    parser_infer = subparsers.add_parser("infer")
    parser_infer.add_argument("--weights", "-w", nargs="?", default="yolov4.weights")
    parser_infer.add_argument("--classes", "-c", nargs="?", default="coco-labels.txt")
    parser_infer.add_argument("--output", "-o")
    parser_infer.add_argument("image")

    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("file_root")
    parser_train.add_argument("annotations")
    parser_train.add_argument("--batch_size", "-b", default=8, type=int)
    parser_train.add_argument("--epochs", "-e", default=5, type=int)
    parser_train.add_argument("--steps", "-s", default=1000, type=int)
    parser_train.add_argument("--output", "-o", default="output.h5")
    parser_train.add_argument("--start_weights", "-w", default=None)
    parser_train.add_argument("--dali_use_gpu", action="store_true")
    parser_train.add_argument("--multigpu", action="store_true")
    parser_train.add_argument("--log_dir", default=None)
    parser_train.add_argument("--ckpt_dir", default=None)
    parser_train.add_argument("--seed", default=None, type=int)

    parser_eval = subparsers.add_parser("eval")
    parser_eval.add_argument("file_root")
    parser_eval.add_argument("annotations")
    parser_eval.add_argument("--weights", "-w", nargs="?", default="yolov4.weights")
    parser_eval.add_argument("--batch_size", "-b", default=1, type=int)
    parser_eval.add_argument("--steps", "-s", default=1000, type=int)

    args = parser.parse_args()

    if args.action == "infer":
        run_infer(args.weights, args.classes, args.image, args.output)
    elif args.action == "train":
        run_training(
            args.file_root, args.annotations, args.batch_size, args.epochs, args.steps,
            output=args.output,
            dali_use_gpu=args.dali_use_gpu,
            log_dir=args.log_dir,
            ckpt_dir=args.ckpt_dir,
            start_weights=args.start_weights,
            multigpu=args.multigpu,
            seed=args.seed
        )
    elif args.action == "eval":
        run_eval(args.file_root, args.annotations, args.weights, args.batch_size, args.steps)
    else:
        print("The " + args.action + " action is not yet implemented :<")
