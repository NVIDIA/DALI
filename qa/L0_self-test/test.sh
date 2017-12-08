#!/bin/bash
set -e
cd "$( dirname "${BASH_SOURCE[0]}" )"

/opt/ndll/build/ndll/run_tests
