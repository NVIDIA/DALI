#!/bin/bash

apt-get update

# glib-2.0 depends on python2, so reinstall the desired python afterward to make sure defaults are right
apt-get install -y --no-install-recommends glib-2.0
apt-get install -y --no-install-recommends --reinstall python$PYVER python$PYVER-dev

pip install nose numpy==1.11.1 opencv-python==3.1.0

cd /opt/dali/ndll/test/python
nosetests --verbose test_pipeline.py
