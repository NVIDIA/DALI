#!/bin/bash -e

pushd ../..

# Create a dummy .git directory to fool linter
mkdir -p .git

# Run linter
cd build-*$PYV*
make lint

popd
