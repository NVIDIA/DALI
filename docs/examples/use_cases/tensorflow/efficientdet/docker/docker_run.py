#!/usr/bin/env python

import argparse
import os

cwd = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="dali-effdet", help="Image tag")
parser.add_argument("--name", default="dali-effdet-cont", help="Container name")
parser.add_argument(
    "--data",
    default=f"{cwd}/mnt/data_dir",
    help="Mount path for additional data (coco-label.txt, weights)",
)
parser.add_argument("--logs", default=f"{cwd}/mnt/logs_dir", help="Directory for logs")
parser.add_argument(
    "--tfrecord_dir",
    default=f"{cwd}/mnt/tfrecord_dir",
    help="Output directory for tfrecord files",
)
parser.add_argument("--cmd", default="", help="Comand for train.py")
parser.add_argument(
    "--coco_train_dataset",
    default="train2017",
    help="Name of COCO training DS. Must have tfrecords created",
)
parser.add_argument(
    "--coco_eval_dataset",
    default="val2017",
    help="Name of COCO validation DS. Must have tfrecords created",
)

args = parser.parse_args()

os.system(
    f"""
    docker run --gpus all \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    --rm \
    -d \
    -v {args.logs}:/dlogs \
    -v {args.data}:/data \
    -v {args.tfrecord_dir}:/tfrec \
    --name {args.name} {args.tag} bash -c \"\
        python3 train.py {args.cmd} \
            --log_dir /dlogs \
            --ckpt_dir /data/ckpt \
            --train_file_pattern '/tfrec/{args.coco_train_dataset}_tf*.tfrecord' \
            --eval_file_pattern '/tfrec/{args.coco_eval_dataset}_tf*.tfrecord' \
            --eval_after_training \
            2>&1 \
        | tee /dlogs/docker_output.log
\""""
)
os.system(f"docker container logs -f -t {args.name}")
