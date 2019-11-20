#!/bin/bash
set -e

conda install --yes -c conda-forge 'protobuf >=3.5'
conda install --yes -c conda-forge 'libjpeg-turbo >=1.5'
conda install --yes -c conda-forge 'ffmpeg >=4.1'
conda install --yes -c conda-forge 'opencv >=3.4.1'
conda install --yes -c conda-forge 'libsndfile >=1.0'
conda install --yes -c conda-forge 'libtiff >=1.0'
