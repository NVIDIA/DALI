# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import os
import re
import sys
import platform
import fnmatch
from distutils.version import StrictVersion

# Find file matching `pattern` in `path`
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result

# Get path to python module `module_name`
def get_module_path(module_name):
    module_path = ''
    for d in sys.path:
        possible_path = os.path.join(d, module_name)
        # skip current dir as this is plugin dir
        if os.path.isdir(possible_path) and len(d) != 0:
            module_path = possible_path
            break
    return module_path

# Get compiler version used to build tensorflow
def get_tf_compiler_version():
    tensorflow_libs = find('libtensorflow_framework*so*', get_module_path('tensorflow'))
    if not tensorflow_libs:
        tensorflow_libs = find('libtensorflow_framework*so*', get_module_path('tensorflow_core'))
        if not tensorflow_libs:
            return ''
    lib = tensorflow_libs[0]
    cmd = 'strings -a ' + lib + ' | grep "GCC: ("'
    s = str(subprocess.check_output(cmd, shell=True))
    lines = s.split('\\n')
    ret_ver = ''
    for line in lines:
        res = re.search("GCC:\s*\(.*\)\s*(\d+.\d+).\d+", line)
        if res:
            ver = res.group(1)
            if not ret_ver or StrictVersion(ret_ver) < StrictVersion(ver):
                ret_ver = ver
    return ret_ver

# Get current tensorflow version
def get_tf_version():
    try:
        import pkg_resources
        s = pkg_resources.get_distribution("tensorflow-gpu").version
    except:
        # pkg_resources.get_distribution doesn't work well with conda installed packages
        try:
            import tensorflow as tf
            s = tf.__version__
        except:
            return ""
    version = re.search("(\d+.\d+).\d+", s).group(1)
    return version

# Get C++ compiler
def get_cpp_compiler():
    return os.environ.get('CXX') or 'g++'

# Get C++ compiler version
def get_cpp_compiler_version():
    cmd = get_cpp_compiler() + ' --version | head -1 | grep "[c|g]++ ("'
    s = str(subprocess.check_output(cmd, shell=True).strip())
    version = re.search("[g|c]\+\+\s*\(.*\)\s*(\d+.\d+).\d+", s).group(1)
    return version

# Runs `which` program
def which(program):
    try:
        return subprocess.check_output('which ' + program, shell=True).strip()
    except:
        return None

# Checks whether we are inside a conda env
def is_conda_env():
    return True if os.environ.get('CONDA_PREFIX') else False

# Get compile and link flags for installed tensorflow
def get_tf_build_flags():
    tf_cflags = ''
    tf_lflags = ''
    try:
        import tensorflow as tensorflow
        tf_cflags=" ".join(tensorflow.sysconfig.get_compile_flags())
        tf_lflags=" ".join(tensorflow.sysconfig.get_link_flags())
    except:
        tensorflow_path = get_module_path('tensorflow')
        if tensorflow_path != '':
            tf_cflags=" ".join(["-I" + tensorflow_path + "/include",  "-I" + tensorflow_path + "/include/external/nsync/public", "-D_GLIBCXX_USE_CXX11_ABI=0"])
            tf_lflags=" ".join(["-L" + tensorflow_path, "-ltensorflow_framework"])

    if tf_cflags == '' and tf_lflags == '':
        raise ImportError('Could not find Tensorflow. Tensorflow must be installed before installing NVIDIA DALI TF plugin')
    return (tf_cflags, tf_lflags)

# Get compile and link flags for installed DALI
def get_dali_build_flags():
    dali_cflags = ''
    dali_lflags = ''
    try:
        import nvidia.dali.sysconfig as dali_sc
        dali_lib_path = dali_sc.get_lib_dir()
        # We are linking with DALI's C library, so we don't need the C++ compile flags
        # including the CXX11_ABI setting
        dali_cflags=" ".join(dali_sc.get_include_flags())
        dali_lflags=" ".join(dali_sc.get_link_flags())
    except:
        dali_path = get_module_path('nvidia/dali')
        if dali_path != '':
            dali_cflags=" ".join(["-I" + dali_path + "/include"])
            dali_lflags=" ".join(["-L" + dali_path, "-ldali"])
    if dali_cflags == '' and dali_lflags == '':
        raise ImportError('Could not find DALI.')
    return (dali_cflags, dali_lflags)

# Get compile and link flags for installed CUDA
def get_cuda_build_flags():
    cuda_cflags = ''
    cuda_lflags = ''
    cuda_home = os.environ.get('CUDA_HOME')
    if not cuda_home:
        cuda_home = '/usr/local/cuda'
    cuda_cflags=" ".join(["-I" + cuda_home + "/include"])
    cuda_lflags=" ".join([])
    return (cuda_cflags, cuda_lflags)

def find_available_prebuilt_tf(requested_version, available_libs):
    req_ver_first, req_ver_second = [int(v) for v in requested_version.split('.', 2)]
    selected_ver = None
    for file in available_libs:
        re_match = re.search(".*(\d+)_(\d+).*", file)
        if re_match is None:
            continue
        ver_first, ver_second = [int(v) for v in re_match.groups()]
        if ver_first == req_ver_first:
            if ver_second <= req_ver_second and (selected_ver is None or selected_ver < (ver_first, ver_second)):
                selected_ver = (ver_first, ver_second)
    return '.'.join([str(v) for v in selected_ver]) if selected_ver is not None else None
