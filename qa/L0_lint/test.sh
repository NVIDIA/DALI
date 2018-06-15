#!/bin/bash
set -e

cd /opt/ndll/build*$PYV*
make lint
