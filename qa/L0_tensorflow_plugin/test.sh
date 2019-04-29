#!/bin/bash -e
# used pip packages
pip_packages="nose tensorflow-gpu"

pushd ../..

source qa/setup_test.sh
cd dali/test/python

# Apparently gcc/g++ installation is broken in the docker image
if ( ! test `find /usr/lib/gcc -name stddef.h` ); then
    apt-get purge --autoremove -y build-essential g++ gcc libc6-dev
    apt-get update && apt-get install -y build-essential g++ gcc libc6-dev
fi

test_body() {
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
}

source ../../../qa/test_template.sh

popd
