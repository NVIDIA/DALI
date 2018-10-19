#!/bin/bash -e

pushd ../..

# Create a dummy .git directory to fool linter
mkdir -p .git

# Run linter
cd build-docker-release
make lint

popd
