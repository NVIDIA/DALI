#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} scikit-build ninja cmake opencv-python-headless'
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

# Note: To link with system ffmpeg, do:
# apt install nasm ffmpeg libavfilter-dev libavformat-dev \
#             libavcodec-dev libswresample-dev libavutil-dev
test_body() {
    # The package name can be nvidia-dali-video, nvidia-dali-video-weekly or nvidia-dali-video-nightly
    pip uninstall -y `pip list | grep nvidia-dali-video | cut -d " " -f1` || true

    # Installing the video plugin
    pip install -v ../../../nvidia_dali_video*.tar.gz

    # Check that the plugin can be loaded
    ${python_invoke_test} test_dali_video_plugin.py:TestDaliVideoPluginLoadOk

    ${python_new_invoke_test} -s . test_dali_video_plugin_decoder
}

pushd ../..
source ./qa/test_template.sh
popd
