#!/bin/bash -e

# Create a dummy .git directory to fool linter
cd /opt/dali
mkdir -p .git

cd /opt/dali/build-*$PYV*
make lint
