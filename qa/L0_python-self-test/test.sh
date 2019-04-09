#!/bin/bash -e
# used pip packages
pip_packages="nose numpy opencv-python tensorflow-gpu torchvision mxnet-cu##CUDA_VERSION##"

pushd ../..

cd dali/test/python

test_body() {
    # test code
    nosetests --verbose test_backend_impl.py
    nosetests --verbose test_pipeline.py
    nosetests --verbose test_plugin_manager.py

    python test_detection_pipeline.py -i 300
    python test_RN50_data_pipeline.py -i 10
    python test_RN50_data_fw_iterators.py -i 30 -b 13

    # DALI TF tests

    # Load plugin OK (supported version of TF in the prebuilt plugins)
    nosetests --verbose test_dali_tf_plugin.py:TestDaliTfPluginLoadOk

    # Manually removing the supported plugin so that it fails
    lib_dir=$(python -c 'import nvidia.dali.sysconfig as sc; print(sc.get_lib_dir())')
    pushd $lib_dir/plugin
    mkdir tmp
    mv *.so tmp
    popd
    nosetests --verbose test_dali_tf_plugin.py:TestDaliTfPluginLoadFail

    # Installing "current" dali tf (built against installed TF)
    pip install ../../../nvidia-dali-tf-plugin*.tar.gz
    nosetests --verbose test_dali_tf_plugin.py:TestDaliTfPluginLoadOk

    # Restore plugins (still works)
    mv $lib_dir/plugin/tmp/* $lib_dir/plugin
    rm -rf $lib_dir/plugin/tmp
    nosetests --verbose test_dali_tf_plugin.py:TestDaliTfPluginLoadOk

    # end DALI TF tests
}

source ../../../qa/test_template.sh

popd
