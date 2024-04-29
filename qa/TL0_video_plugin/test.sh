#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package}'
target_dir=./dali/test/python

# reduce the lenght of the sanitizers tests as much as possible
# use only one TF verion, don't test virtual env
if [ -n "$DALI_ENABLE_SANITIZERS" ]; then
    one_config_only=true
else
    # populate epilog and prolog with variants to enable/disable conda and virtual env
    # every test will be executed for bellow configs
    prolog=(: enable_virtualenv)
    epilog=(: disable_virtualenv)
fi

test_body() {
    # The package name can be nvidia-dali-video, nvidia-dali-video-weekly or nvidia-dali-video-nightly
    pip uninstall -y `pip list | grep nvidia-dali-video | cut -d " " -f1` || true

    # No plugin installed, should fail
    ${python_invoke_test} test_dali_video_plugin.py:TestDaliVideoPluginLoadFail

    # Installing the video plugin
    pip install ../../../nvidia-dali-video*.tar.gz

    ${python_invoke_test} test_dali_video_plugin.py:TestDaliVideoPluginLoadOk
}

pushd ../..
source ./qa/test_template.sh
popd
