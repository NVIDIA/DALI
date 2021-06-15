#!/usr/bin/env python

import argparse
import os

cwd = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="dali-effdet", help="Image tag")
parser.add_argument("--name", default="dali-effdet-gentf", help="Container name")
parser.add_argument("--coco", help="Mount path for dataset")
parser.add_argument(
    "--tfrecord_dir",
    default=f"{cwd}/mnt/tfrecord_dir",
    help="Output directory for tfrecord files",
)
parser.add_argument(
    "--coco_dataset", default=f"train2017", help="Coco dataset to convert"
)
args = parser.parse_args()


os.system(
    f"""
    docker run --gpus all --rm \
    -v {args.tfrecord_dir}:/tfrec \
    -v {args.coco}:/coco \
    --name {args.name} {args.tag} bash -c \"\
        cd dataset && \
        python3 create_coco_tfrecord.py \
            --image_dir /coco/{args.coco_dataset} \
            --caption_annotations_file /coco/annotations/captions_{args.coco_dataset}.json \
            --object_annotations_file /coco/annotations/instances_{args.coco_dataset}.json \
            --output_file_prefix /tfrec/{args.coco_dataset}_tf && \
        python3 create_tfrecord_indexes.py \
            --tfrecord_file_pattern '/tfrec/{args.coco_dataset}_tf*.tfrecord' \
            --tfrecord2idx_script '../__docker_tfrecord2idx' \
\""""
)
