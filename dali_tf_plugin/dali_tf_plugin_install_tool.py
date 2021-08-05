# Copyright (c) 2019-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import platform
from shutil import copyfile
from dali_tf_plugin_utils import *
import os
from distutils.version import StrictVersion
from pathlib import Path
import tempfile
from stubgen import stubgen

class InstallerHelper:
    def __init__(self, plugin_dest_dir = None):
        self.src_path = os.path.dirname(os.path.realpath(__file__))
        self.dali_lib_path = get_module_path('nvidia/dali')
        self.tf_path = get_module_path('tensorflow')
        self.plugin_dest_dir = os.path.join(self.src_path, 'nvidia', 'dali_tf_plugin') if plugin_dest_dir is None else plugin_dest_dir
        self.is_conda = is_conda_env()
        self.tf_version = get_tf_version()
        self.tf_compiler = get_tf_compiler_version()
        self.cpp_compiler = get_cpp_compiler()
        self.default_cpp_version = get_cpp_compiler_version()
        self.alt_compiler = 'g++-{}'.format(self.tf_compiler)
        self.has_alt_compiler = which(self.alt_compiler) is not None
        self.platform_system = platform.system()
        self.platform_machine = platform.machine()
        self.is_compatible_with_prebuilt_bin = self.platform_system == 'Linux' and self.platform_machine == 'x86_64'
        self.prebuilt_dir = os.path.join(self.src_path, 'prebuilt')
        self.prebuilt_stub_dir = os.path.join(self.prebuilt_dir, 'stub')
        dali_stubs = find('libdali.so', self.prebuilt_stub_dir)
        self.prebuilt_dali_stub = dali_stubs[0] if len(dali_stubs) > 0 else None

        # If set, checking for prebuilt binaries or compiler version check is disabled
        self.always_build = bool(int(os.getenv('DALI_TF_ALWAYS_BUILD', '0')))

        # Can install prebuilt if both conditions apply:
        # - we know the compiler used to build TF
        # - we have prebuilt artifacts for that compiler version
        # - We have an exact match with the TF version major.minor or an exact match of the
        #   major version and the minor version in the prebuilt plugin is lower than the requested one.
        self.can_install_prebuilt = not self.always_build and \
            bool(self.tf_compiler) and \
            StrictVersion(self.tf_compiler) >= StrictVersion('5.0') and \
            self.is_compatible_with_prebuilt_bin and \
            self.prebuilt_dali_stub is not None

        self.prebuilt_plugins_available = []
        self.prebuilt_plugin_best_match = None
        self.plugin_name = None

        if self.can_install_prebuilt:
            self.prebuilt_plugins_available = find('libdali_tf_*.so', self.prebuilt_dir)
            best_version = find_available_prebuilt_tf(self.tf_version, self.prebuilt_plugins_available)
            if best_version is None:
                # No prebuilt plugins available
                self.can_install_prebuilt = False
            else:
                tf_version_underscore = best_version.replace('.', '_')
                self.plugin_name = 'libdali_tf_' + tf_version_underscore + '.so'
                self.prebuilt_plugin_best_match = os.path.join(self.prebuilt_dir, self.plugin_name)

        # Allow to compile if either condition apply
        # - The default C++ compiler version matches the one used to build TF
        # - The compiler used to build TF is unknown
        # - Both TF and default compilers are >= 5.0
        self.can_default_compile = self.always_build or \
            self.default_cpp_version == self.tf_compiler or \
            not bool(self.tf_compiler) or \
            (StrictVersion(self.default_cpp_version) >= StrictVersion('5.0') and
                StrictVersion(self.tf_compiler) >= StrictVersion('5.0'))

    def debug_str(self):
        s = "\n Environment:"
        s += "\n ---------------------------------------------------------------------------------------------------------"
        s += "\n Platform system:                      {}".format(self.platform_system)
        s += "\n Platform machine:                     {}".format(self.platform_machine)
        s += "\n DALI lib path:                        {}".format(self.dali_lib_path or "Not Installed")
        s += "\n TF path:                              {}".format(self.tf_path or "Not Installed")
        s += "\n DALI TF plugin destination directory: {}".format(self.plugin_dest_dir)
        s += "\n Is Conda environment?                 {}".format("Yes" if self.is_conda else "No")
        s += "\n Using compiler:                       \"{}\", version {}".format(self.cpp_compiler, self.default_cpp_version or "Unknown")
        s += "\n TF version installed:                 {}".format(self.tf_version or "Unknown")
        if self.tf_version:
            s += "\n g++ version used to compile TF:       {}".format(self.tf_compiler or "Unknown")
            s += "\n Is {} present in the system?     {}".format(self.alt_compiler, "Yes" if self.has_alt_compiler else "No")
            s += "\n Can install prebuilt plugin?          {}".format("Yes" if self.can_install_prebuilt else "No")
            s += "\n Prebuilt plugin path:                 {}".format(self.prebuilt_plugin_best_match or "N/A")
            s += "\n Prebuilt plugins available:           {}".format(", ".join(self.prebuilt_plugins_available) or "N/A")
            s += "\n Prebuilt DALI stub available:         {}".format(self.prebuilt_dali_stub or "N/A")
            s += "\n Can compile with default compiler?    {}".format("Yes" if self.can_default_compile else "No")
            s += "\n Can compile with alt compiler?        {}".format("Yes" if self.has_alt_compiler else "No")
        s += "\n---------------------------------------------------------------------------------------------------------"

        return s

    def check_import(self, lib_path, dali_stub):
        import tensorflow as tf
        lib_name = os.path.basename(lib_path)
        dali_stub_name = os.path.basename(dali_stub)
        print("Importing the TF library to check for errors")

        # The DALI TF lib and the DALI stub lib should be at the same directory for check_import to succeed
        # Unfortunately the copy is necessary because we can't change LD_LIBRARY_PATH from within the script
        with tempfile.TemporaryDirectory(prefix="check_import_tmp") as tmpdir:
            lib_path_tmpdir = os.path.join(tmpdir, lib_name)
            copyfile(lib_path, lib_path_tmpdir)

            dali_stub_tmpdir = os.path.join(tmpdir, dali_stub_name)
            copyfile(dali_stub, dali_stub_tmpdir)

            try:
                print("Loading DALI TF library: ", lib_path_tmpdir)
                tf.load_op_library(lib_path_tmpdir)
                return True
            except Exception as e:
                print("Failed to import TF library: ", str(e))
                return False

    def install_prebuilt(self):
        assert(self.can_install_prebuilt)
        assert(self.prebuilt_plugin_best_match is not None)
        assert(self.plugin_name is not None)
        print("Tensorflow was built with g++ {}, providing prebuilt plugin".format(self.tf_compiler))

        if self.check_import(self.prebuilt_plugin_best_match, self.prebuilt_dali_stub):
            print("Copy {} to {}".format(self.prebuilt_plugin_best_match, self.plugin_dest_dir))
            plugin_dest =  os.path.join(self.plugin_dest_dir, self.plugin_name)
            copyfile(self.prebuilt_plugin_best_match, plugin_dest)
            return True
        else:
            print(f"Error importing {self.prebuilt_plugin_best_match}, will not install prebuilt plugin")
            return False

    def install(self):
        print("Checking build environment for DALI TF plugin ...")
        print(self.debug_str())

        Path(self.plugin_dest_dir).mkdir(parents=True, exist_ok=True)

        if not self.tf_version or not self.tf_path:
            error_msg = "Installation error:"
            error_msg += "\n Tensorflow installation not found. Install `tensorflow-gpu` and try again"
            error_msg += '\n' + self.debug_str()
            raise ImportError(error_msg)

        compiler = self.cpp_compiler

        # From tensorflow team (https://github.com/tensorflow/tensorflow/issues/29643):
        # Our pip packages are still built with gcc 4.8."
        # To make anything that uses C++ APIs work, all custom ops need to be built
        # with the same compiler (and the version) we use to build the pip packages.
        # Anything not built with that may break due to compilers generating ABIs differently."

        # Note: https://github.com/tensorflow/custom-op
        # Packages are also built for gcc 5.4 now, so we are also providing prebuilt plugins for 5.4
        if self.can_install_prebuilt:
            if self.install_prebuilt():
                return
            else:
                print("Installation of prebuilt plugins failed, will try building from source")

        if not self.can_default_compile:
            if self.has_alt_compiler:
                print("Will use alternative compiler {}".format(self.alt_compiler))
                compiler = self.alt_compiler
            elif self.is_conda:
                error_msg = "Installation error:"
                error_msg += "\n Conda C++ compiler version should be the same as the compiler used to build tensorflow ({} != {}).".format(self.default_cpp_version, self.tf_compiler)
                error_msg += "\n Try to run `conda install gxx_linux-64=={}` or install an alternative compiler `g++-{}` and install again".format(self.tf_compiler, self.tf_compiler)
                error_msg += '\n' + self.debug_str()
                raise ImportError(error_msg)
            else:
                error_msg = "Installation error:"
                error_msg += "\n Tensorflow was built with a different compiler than the currently installed ({} != {})".format(self.default_cpp_version, self.tf_compiler)
                error_msg += "\n Try to install `g++-{}` or use CXX environment variable to point to the right compiler and install again".format(self.tf_compiler)
                error_msg += '\n' + self.debug_str()
                raise ImportError(error_msg)

        print("Proceed with build...")
        cuda_cflags, cuda_lflags = get_cuda_build_flags()

        with tempfile.TemporaryDirectory(prefix="dali_stub_") as tmpdir:
            # Building a DALI stub library. During runtime, the real libdali.so will be already loaded at the moment when the DALI TF plugin is loaded
            # This is done to avoid depending on DALI being installed during DALI TF sdist installation
            dali_stub_src = os.path.join(tmpdir, 'dali_stub.cc')
            dali_stub_lib = os.path.join(tmpdir, 'libdali.so')
            dali_c_api_hdr = os.path.join(self.src_path, 'include', 'dali', 'c_api.h')
            with open(dali_stub_src, 'w+') as f:
                stubgen(header_filepath=dali_c_api_hdr, out_file=f)

            dali_lflags = '-L' + tmpdir + ' -ldali'
            dali_cflags = '-I' + os.path.join(self.src_path, 'include')

            cmd = compiler + ' -Wl,-R,\'$ORIGIN/..\' -std=c++14 -DNDEBUG -shared ' \
                + dali_stub_src + ' -o ' + dali_stub_lib + ' -fPIC ' + dali_cflags + ' ' \
                + cuda_cflags + ' ' + cuda_lflags + ' -O2'
            print('Building DALI stub lib:\n\n ' + cmd + '\n\n')
            subprocess.check_call(cmd, cwd=self.src_path, shell=True)

            tf_cflags, tf_lflags = get_tf_build_flags()

            filenames = ['daliop.cc', 'dali_dataset_op.cc']
            plugin_src = ''
            for filename in filenames:
                plugin_src = plugin_src + ' ' + os.path.join(self.src_path, filename)

            lib_filename = 'libdali_tf_current.so'
            lib_path =  os.path.join(self.plugin_dest_dir, lib_filename)

            # Note: DNDEBUG flag is needed due to issue with TensorFlow custom ops:
            # https://github.com/tensorflow/tensorflow/issues/17316
            # Do not remove it.
            cmd = compiler + ' -Wl,-R,\'$ORIGIN/..\' -Wl,-rpath,\'$ORIGIN\' -std=c++14 -DNDEBUG -shared ' \
                + plugin_src + ' -o ' + lib_path + ' -fPIC ' + dali_cflags + ' ' \
                + tf_cflags + ' ' + cuda_cflags + ' ' + dali_lflags + ' ' + tf_lflags + ' ' \
                + cuda_lflags + ' -O2'
            print("Build DALI TF library:\n\n " + cmd + '\n\n')
            subprocess.check_call(cmd, cwd=self.src_path, shell=True)

            if not self.check_import(lib_path, dali_stub_lib):
                raise ImportError("Error while importing the DALI TF plugin built from source, will not install")
            print("Installation successful")

def main():
    env = InstallerHelper()
    env.install()

if __name__ == "__main__":
    main()
