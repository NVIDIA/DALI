#!/bin/bash -e
# used pip packages

pip_packages='tensorflow-gpu'

test_body() {
    # test code
    echo "---------Testing TENSORFLOW;DALI----------"
    ( set -x && python -c "import tensorflow; import nvidia.dali.plugin.tf as dali_tf; daliop = dali_tf.DALIIterator()" )
    echo "---------Testing DALI;TENSORFLOW----------"
    ( set -x && python -c "import nvidia.dali.plugin.tf as dali_tf; import tensorflow; daliop = dali_tf.DALIIterator()" )
}

pushd ../../
source ./qa/test_template.sh
popd
