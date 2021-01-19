#!/bin/bash -e
# used pip packages
pip_packages="nose tensorflow-gpu"
target_dir=./dali/test/python

# populate epilog and prolog with variants to enable/disable conda and virtual env
# every test will be executed for bellow configs
prolog=(: enable_virtualenv)
epilog=(: disable_virtualenv)

test_body() {
    # The package name can be nvidia-dali-tf-plugin,  nvidia-dali-tf-plugin-weekly or  nvidia-dali-tf-plugin-nightly
    pip uninstall -y `pip list | grep nvidia-dali-tf-plugin | cut -d " " -f1` || true

    # No plugin installed, should fail
    nosetests --verbose test_dali_tf_plugin.py:TestDaliTfPluginLoadFail

    # Installing "current" dali tf (built against installed TF)
    pip install ../../../nvidia-dali-tf-plugin*.tar.gz
    nosetests --verbose test_dali_tf_plugin.py:TestDaliTfPluginLoadOk

    # DALI TF run
    nosetests --verbose test_dali_tf_plugin_run.py

    # DALI TF DATASET run
    nosetests --verbose test_dali_tf_dataset.py
    nosetests --verbose test_dali_tf_dataset_shape.py
    nosetests --verbose test_dali_tf_dataset_eager.py
    nosetests --verbose test_dali_tf_dataset_graph.py    
}

pushd ../..
source ./qa/test_template.sh
popd
