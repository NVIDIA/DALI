#!/bin/bash -e
# used pip packages
pip_packages='jupyter numpy matplotlib jax flax'
target_dir=./docs/examples

test_body() {
    test_files=(
        "frameworks/jax/jax-basic_example.ipynb"
    )

    # JAX no longer releases for Python3.8, the last available version
    # does not support newer dlpack protocol we use for jax_function
    PY_VERSION=$(python -c "import sys; print(\"{}.{}\".format(sys.version_info[0],sys.version_info[1]))")
    PY_VERSION_SHORT=${PY_VERSION/\./}
    if [ "$PY_VERSION_SHORT" -ge 39 ]; then
        test_files+=( "custom_operations/jax_operator_basic.ipynb" )
    fi

    for f in ${test_files[@]}; do
        jupyter nbconvert --to notebook --inplace --execute \
                        --ExecutePreprocessor.kernel_name=python${PYVER:0:1} \
                        --ExecutePreprocessor.timeout=300 $f;
    done
}

pushd ../..
source ./qa/test_template.sh
popd
