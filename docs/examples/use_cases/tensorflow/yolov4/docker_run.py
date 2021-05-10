#!/usr/bin/env python

import argparse
import os

cwd = os.getcwd()

parser = argparse.ArgumentParser()
parser.add_argument('--tag', default="dali-yolov4", help='Image tag')
parser.add_argument('--name', default="dali-yolov4-cont", help="Container name")
parser.add_argument('--coco', default=f"{cwd}/mnt/coco_dir", help="Mount path for dataset")
parser.add_argument('--data', default=f"{cwd}/mnt/data_dir", help="Mount path for additional data (coco-label.txt, weights)")
parser.add_argument('--logs', default=f"{cwd}/mnt/logs_dir", help="Directory for logs")
parser.add_argument('--cmd', default="", help="Comand for main.py")
args = parser.parse_args()

os.system(f"docker run --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --rm -v {args.logs}:/dlogs -v {args.coco}:/coco -v {args.data}:/data --name {args.name} {args.tag} bash -c \"python3 main.py {args.cmd} 2>&1 | tee -a /dlogs/output.log\"")
