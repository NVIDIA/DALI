#!/bin/bash

pip install nose

cd /opt/dali/ndll/test/python
nosetests --verbose test_pipeline.py
