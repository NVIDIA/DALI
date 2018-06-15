#!/bin/bash

pip install nose numpy==1.11.1 opencv-python==3.1.0

cd /opt/dali/ndll/test/python
nosetests --verbose test_pipeline.py
