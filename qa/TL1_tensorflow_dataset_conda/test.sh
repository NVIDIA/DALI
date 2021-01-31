#!/bin/bash -e
# used pip packages
# use TF that is installed from conda when DALI is installed
pip_packages="nose jupyter"
target_dir=./dali/test/python

# populate epilog and prolog with variants to enable/disable conda
# every test will be executed for bellow configs
prolog=(enable_conda)
epilog=(disable_conda)

test_body() {

    is_compatible=$(python -c 'import nvidia.dali.plugin.tf as dali_tf; print(dali_tf.dataset_compatible_tensorflow())')
    if [ $is_compatible = 'True' ]; then
        # DALI TF DATASET run
        nosetests --verbose -s test_dali_tf_dataset_graph.py:_test_tf_dataset_other_gpu
        nosetests --verbose -s test_dali_tf_dataset_graph.py:_test_tf_dataset_multigpu_manual_placement
        nosetests --verbose -s test_dali_tf_dataset_eager.py:_test_tf_dataset_other_gpu
        nosetests --verbose -s test_dali_tf_dataset_eager.py:_test_tf_dataset_multigpu_manual_placement
        nosetests --verbose -s test_dali_tf_dataset_eager.py:_test_tf_dataset_multigpu_mirrored_strategy
        nosetests --verbose -s test_dali_tf_dataset_mnist_eager.py
        nosetests --verbose -s test_dali_tf_dataset_mnist_graph.py

        # DALI TF Notebooks run
        pushd ../../../docs/examples/frameworks/tensorflow/
        jupyter nbconvert tensorflow-dataset.ipynb \
                  --to notebook --inplace --execute \
                  --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                  --ExecutePreprocessor.timeout=600

        is_compatible_distributed=$(python -c 'import nvidia.dali.plugin.tf as dali_tf; print(dali_tf.dataset_distributed_compatible_tensorflow())')
        if [ $is_compatible_distributed = 'True' ]; then
            jupyter nbconvert tensorflow-dataset-multigpu.ipynb \
                    --to notebook --inplace --execute \
                    --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                    --ExecutePreprocessor.timeout=600
        fi
        
        popd
    fi
}

pushd ../..
source ./qa/test_template.sh
popd
