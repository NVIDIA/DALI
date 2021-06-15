#!/usr/bin/env python

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--tag", default="dali-effdet", help="Tag for ")
args = parser.parse_args()

# We copy tfrecord2idx to
os.system(" cp ../../../../../../tools/tfrecord2idx ../__docker_tfrecord2idx")
os.system(
    f"cd .. && docker build --network=internal -t {args.tag} -f docker/Dockerfile ."
)
os.system("rm -rf ../__docker_tfrecord2idx")
