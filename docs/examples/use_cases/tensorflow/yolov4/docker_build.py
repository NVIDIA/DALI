#!/usr/bin/env python

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--tag', default="dali-yolov4", help='Tag for ')
args = parser.parse_args()

os.system(f"docker build -t {args.tag} .")
