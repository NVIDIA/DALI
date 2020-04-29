#!/bin/bash -e
# used pip packages
# use TF that is installed from conda when DALI is installed
pip_packages="nose"
target_dir=./dali/test/python

# populate epilog and prolog with variants to enable/disable conda
# every test will be executed for bellow configs
prolog=(enable_conda)
epilog=(disable_conda)

test_body() {
    nosetests --verbose test_dali_tf_plugin.py:TestDaliTfPluginLoadOk

    # DALI TF run
    nosetests --verbose test_dali_tf_plugin_run.py

    # DALI TF DATASET run
    nosetests --verbose test_dali_tf_dataset.py

    nosetests --verbose test_dali_tf_dataset_shape.py
}

pushd ../..
source ./qa/test_template.sh
popd
