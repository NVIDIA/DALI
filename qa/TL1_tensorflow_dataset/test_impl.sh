#!/bin/bash -e
# used pip packages
pip_packages='${python_test_runner_package} jupyter tensorflow-gpu'
target_dir=./dali/test/python

test_body() {
    # The package name can be nvidia-dali-tf-plugin,  nvidia-dali-tf-plugin-weekly or  nvidia-dali-tf-plugin-nightly
    pip uninstall -y `pip list | grep nvidia-dali-tf-plugin | cut -d " " -f1` || true

    # Installing "current" dali tf (built against installed TF)
    pip install ../../../nvidia-dali-tf-plugin*.tar.gz

    is_compatible=$(python -c 'import nvidia.dali.plugin.tf as dali_tf; print(dali_tf.dataset_compatible_tensorflow())')
    if [ $is_compatible = 'True' ]; then
        # DALI TF DATASET run
        ${python_invoke_test} test_dali_tf_dataset_graph.py:_test_tf_dataset_other_gpu
        ${python_invoke_test} test_dali_tf_dataset_graph.py:_test_tf_dataset_multigpu_manual_placement
        ${python_invoke_test} test_dali_tf_dataset_eager.py:_test_tf_dataset_other_gpu
        ${python_invoke_test} test_dali_tf_dataset_eager.py:_test_tf_dataset_multigpu_manual_placement
        ${python_invoke_test} test_dali_tf_dataset_eager.py:_test_tf_dataset_multigpu_mirrored_strategy
        ${python_invoke_test} test_dali_tf_dataset_mnist_eager.py
        ${python_invoke_test} test_dali_tf_dataset_mnist_graph.py

        # DALI TF Notebooks run
        pushd ../../../docs/examples/frameworks/tensorflow/
        jupyter nbconvert tensorflow-dataset.ipynb \
                  --to notebook --inplace --execute \
                  --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                  --ExecutePreprocessor.timeout=600

        # due to compatibility problems between the driver, cuda version and
        # TensorFlow 2.12 test_keras_multi_gpu_mirrored_strategy doesn't work.
        is_compatible_distributed=$(python -c 'import nvidia.dali.plugin.tf as dali_tf; \
                                               import tensorflow as tf; \
                                               from distutils.version import LooseVersion; \
                                               print(dali_tf.dataset_distributed_compatible_tensorflow() \
                                               and LooseVersion(tf.__version__) < LooseVersion("2.12.0"))')
        if [ $is_compatible_distributed = 'True' ]; then
            jupyter nbconvert tensorflow-dataset-multigpu.ipynb \
                    --to notebook --inplace --execute \
                    --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                    --ExecutePreprocessor.timeout=600
        fi

        popd
    fi
}
