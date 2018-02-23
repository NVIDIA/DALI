#!/bin/bash

pip install nose

cd /opt/ndll/ndll/test/python
nosetests --verbose test_pipeline.py
