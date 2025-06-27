# Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from dali_tf_plugin_utils import (
    get_module_path,
    is_conda_env,
    get_tf_version,
    get_tf_compiler_version,
    get_cpp_compiler,
    get_cpp_compiler_version,
    which,
    find,
    find_available_prebuilt_tf,
    get_cuda_build_flags,
    get_tf_build_flags,
)
import os
from packaging.version import Version
from pathlib import Path
import tempfile
from stubgen import stubgen
from multiprocessing import Process
import subprocess  # nosec B404


def plugin_load_and_test(dali_tf_path):
    # Make sure that TF won't try using CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import tensorflow as tf

    try:
        from tensorflow.compat.v1 import Session
    except Exception:
        # Older TF versions don't have compat.v1 layer
        from tensorflow import Session

    try:
        tf.compat.v1.disable_eager_execution()
    except Exception:
        pass

    @pipeline_def()
    def get_dali_pipe():
        data = types.Constant(1)
        return data

    _dali_tf_module = tf.load_op_library(dali_tf_path)
    _dali_tf = _dali_tf_module.dali

    def get_data():
        batch_size = 3
        pipe = get_dali_pipe(batch_size=batch_size, device_id=None, num_threads=1)

        out = []
        with tf.device("/cpu"):
            data = _dali_tf(
                serialized_pipeline=pipe.serialize(),
                shapes=[(batch_size,)],
                dtypes=[tf.int32],
                device_id=None,
                batch_size=batch_size,
                exec_separated=False,
                gpu_prefetch_queue_depth=2,
                cpu_prefetch_queue_depth=2,
            )
            out.append(data)
        return [out]

    test_batch = get_data()
    with Session() as sess:
        for _ in range(3):
            print(sess.run(test_batch))


class InstallerHelper:
    def __init__(self, plugin_dest_dir=None):
        self.src_path = os.path.dirname(os.path.realpath(__file__))
        self.dali_lib_path = get_module_path("nvidia/dali")
        self.tf_path = get_module_path("tensorflow")
        self.plugin_dest_dir = (
            os.path.join(self.src_path, "nvidia", "dali_tf_plugin")
            if plugin_dest_dir is None
            else plugin_dest_dir
        )
        self.is_conda = is_conda_env()
        self.tf_version = get_tf_version()
        self.tf_compiler = get_tf_compiler_version()
        self.cpp_compiler = get_cpp_compiler()
        self.default_cpp_version = get_cpp_compiler_version()
        self.alt_compiler = "g++-{}".format(self.tf_compiler)
        self.has_alt_compiler = which(self.alt_compiler) is not None
        self.platform_system = platform.system()
        self.platform_machine = platform.machine()
        self.is_compatible_with_prebuilt_bin = (
            self.platform_system == "Linux" and self.platform_machine == "x86_64"
        )
        self.prebuilt_dir = os.path.join(self.src_path, "prebuilt")
        self.prebuilt_stub_dir = os.path.join(self.prebuilt_dir, "stub")
        dali_stubs = find("libdali.so", self.prebuilt_stub_dir)
        self.prebuilt_dali_stub = dali_stubs[0] if len(dali_stubs) > 0 else None

        # If set, checking for prebuilt binaries or compiler version check is disabled
        self.always_build = bool(int(os.getenv("DALI_TF_ALWAYS_BUILD", "0")))

        # Can install prebuilt if both conditions apply:
        # - we know the compiler used to build TF
        # - we have prebuilt artifacts for that compiler version
        # - We have an exact match with the TF version major.minor or an exact match of the
        #   major version and the minor version in the prebuilt plugin is lower than the
        #   requested one.
        self.can_install_prebuilt = (
            not self.always_build
            and bool(self.tf_compiler)
            and Version(self.tf_compiler) >= Version("5.0")
            and self.is_compatible_with_prebuilt_bin
            and self.prebuilt_dali_stub is not None
        )

        self.prebuilt_plugins_available = []
        self.prebuilt_plugin_best_match = None
        self.plugin_name = None

        self.prebuilt_exact_ver = False
        if self.can_install_prebuilt:
            self.prebuilt_plugins_available = find("libdali_tf_*.so", self.prebuilt_dir)
            best_version = find_available_prebuilt_tf(
                self.tf_version, self.prebuilt_plugins_available
            )
            if best_version is None:
                # No prebuilt plugins available
                self.can_install_prebuilt = False
            else:
                self.prebuilt_exact_ver = best_version == self.tf_version
                tf_version_underscore = best_version.replace(".", "_")
                self.plugin_name = "libdali_tf_" + tf_version_underscore + ".so"
                self.prebuilt_plugin_best_match = os.path.join(self.prebuilt_dir, self.plugin_name)

        # Allow to compile if either condition apply
        # - The default C++ compiler version matches the one used to build TF
        # - The compiler used to build TF is unknown
        # - Both TF and default compilers are >= 5.0
        self.can_default_compile = (
            self.always_build
            or self.default_cpp_version == self.tf_compiler
            or not bool(self.tf_compiler)
            or (
                Version(self.default_cpp_version) >= Version("5.0")
                and Version(self.tf_compiler) >= Version("5.0")
            )
        )

    def debug_str(self):
        s = "\n Environment:"
        s += "\n ----------------------------------------------------------------------------------"
        s += "\n Platform system:                      {}".format(self.platform_system)
        s += "\n Platform machine:                     {}".format(self.platform_machine)
        s += "\n DALI lib path:                        {}".format(
            self.dali_lib_path or "Not Installed"
        )
        s += "\n TF path:                              {}".format(self.tf_path or "Not Installed")
        s += "\n DALI TF plugin destination directory: {}".format(self.plugin_dest_dir)
        s += "\n Is Conda environment?                 {}".format("Yes" if self.is_conda else "No")
        s += '\n Using compiler:                       "{}", version {}'.format(
            self.cpp_compiler, self.default_cpp_version or "Unknown"
        )
        s += "\n TF version installed:                 {}".format(self.tf_version or "Unknown")
        if self.tf_version:
            s += "\n g++ version used to compile TF:       {}".format(self.tf_compiler or "Unknown")
            s += "\n Is {} present in the system?     {}".format(
                self.alt_compiler, "Yes" if self.has_alt_compiler else "No"
            )
            s += "\n Can install prebuilt plugin?          {}".format(
                "Yes" if self.can_install_prebuilt else "No"
            )
            s += "\n Prebuilt for exact TF version?        {}".format(
                "Yes" if self.prebuilt_exact_ver else "No"
            )
            s += "\n Prebuilt plugin path:                 {}".format(
                self.prebuilt_plugin_best_match or "N/A"
            )
            s += "\n Prebuilt plugins available:           {}".format(
                ", ".join(self.prebuilt_plugins_available) or "N/A"
            )
            s += "\n Prebuilt DALI stub available:         {}".format(
                self.prebuilt_dali_stub or "N/A"
            )
            s += "\n Can compile with default compiler?    {}".format(
                "Yes" if self.can_default_compile else "No"
            )
            s += "\n Can compile with alt compiler?        {}".format(
                "Yes" if self.has_alt_compiler else "No"
            )
        s += "\n-----------------------------------------------------------------------------------"

        return s

    def _test_plugin_in_tmp_dir(self, lib_path, dali_stub, test_fn):
        lib_name = os.path.basename(lib_path)
        dali_stub_name = os.path.basename(dali_stub)
        print("Importing the DALI TF library to check for errors")

        # The DALI TF lib and the DALI stub lib should be at the same directory for
        # check_load_plugin to succeed. Unfortunately the copy is necessary because we
        # can't change LD_LIBRARY_PATH from within the script
        with tempfile.TemporaryDirectory(prefix="check_load_plugin_tmp") as tmpdir:
            lib_path_tmpdir = os.path.join(tmpdir, lib_name)
            copyfile(lib_path, lib_path_tmpdir)

            dali_stub_tmpdir = os.path.join(tmpdir, dali_stub_name)
            copyfile(dali_stub, dali_stub_tmpdir)

            try:
                print("Loading DALI TF library: ", lib_path_tmpdir)
                # try in a separate process just in case it recives SIGV
                p = Process(target=test_fn, args=(lib_path_tmpdir,))
                p.start()
                p.join(10)
                ret = p.exitcode
                if ret is None:
                    p.terminate()
                    p.join()
                if ret != 0:
                    print(f"Failed to import TF library, importing returned {ret}")
                return ret == 0
            except Exception as e:
                print("Failed to import TF library: ", str(e))
                return False

    def check_load_plugin(self, lib_path, dali_stub):
        import tensorflow as tf

        return self._test_plugin_in_tmp_dir(lib_path, dali_stub, tf.load_op_library)

    def test_plugin(self, lib_path, dali_stub):
        return self._test_plugin_in_tmp_dir(lib_path, dali_stub, plugin_load_and_test)

    def check_plugin(self, plugin_path, dali_stub_path):
        dali_available = True
        try:
            import nvidia.dali as dali

            assert dali
        except ImportError:
            dali_available = False

        if dali_available:
            # If DALI is available, test the plugin
            return self.test_plugin(plugin_path, dali_stub_path)
        else:
            # If DALI not available, at least check loading to TF
            return self.check_load_plugin(plugin_path, dali_stub_path)

    def install_prebuilt(self):
        assert self.can_install_prebuilt
        assert self.prebuilt_plugin_best_match is not None
        assert self.plugin_name is not None
        print(f"Tensorflow was built with g++ {self.tf_compiler}, providing prebuilt plugin")

        if self.check_plugin(self.prebuilt_plugin_best_match, self.prebuilt_dali_stub):
            print("Copy {} to {}".format(self.prebuilt_plugin_best_match, self.plugin_dest_dir))
            plugin_dest = os.path.join(self.plugin_dest_dir, self.plugin_name)
            copyfile(self.prebuilt_plugin_best_match, plugin_dest)
            print("Installation successful")
            return True
        else:
            print(
                f"Failed check for {self.prebuilt_plugin_best_match},"
                + "will not install prebuilt plugin"
            )
            return False

    def get_compiler(self):
        compiler = self.cpp_compiler
        if not self.can_default_compile:
            if self.has_alt_compiler:
                print("Will use alternative compiler {}".format(self.alt_compiler))
                compiler = self.alt_compiler
            elif self.is_conda:
                error_msg = "Installation error:"
                error_msg += (
                    "\n Conda C++ compiler version should be the same as the compiler "
                    + "used to build tensorflow "
                    + f"({self.default_cpp_version} != {self.tf_compiler})."
                )
                error_msg += (
                    f"\n Try to run `conda install gxx_linux-64=={self.tf_compiler}` "
                    + f"or install an alternative compiler `g++-{self.tf_compiler}` and "
                    + "install again"
                )
                error_msg += "\n" + self.debug_str()
                raise ImportError(error_msg)
            else:
                error_msg = "Installation error:"
                error_msg += (
                    "\n Tensorflow was built with a different compiler than the "
                    + "currently installed "
                    + f"({self.default_cpp_version} != {self.tf_compiler})"
                )
                error_msg += (
                    f"\n Try to install `g++-{self.tf_compiler}` or use CXX "
                    + "environment variable to point to the right compiler and install again"
                )
                error_msg += "\n" + self.debug_str()
                raise ImportError(error_msg)
        return compiler

    def build(self):
        print("Proceed with build from source...")
        compiler = self.get_compiler()
        cuda_cflags, cuda_lflags = get_cuda_build_flags()

        with tempfile.TemporaryDirectory(prefix="dali_stub_") as tmpdir:
            # Building a DALI stub library. During runtime, the real libdali.so will be already
            # loaded at the moment when the DALI TF plugin is loaded
            # This is done to avoid depending on DALI being installed during
            # DALI TF sdist installation
            dali_stub_src = os.path.join(tmpdir, "dali_stub.cc")
            dali_stub_lib = os.path.join(tmpdir, "libdali.so")
            dali_c_api_hdr = os.path.join(self.src_path, "include", "dali", "dali.h")
            with open(dali_stub_src, "w+") as f:
                stubgen(header_filepath=dali_c_api_hdr, out_file=f)

            dali_lflags = "-L" + tmpdir + " -ldali"
            dali_cflags = "-I" + os.path.join(self.src_path, "include")

            cmd = [compiler, "-Wl,-R,$ORIGIN/..", "-std=c++14", "-DNDEBUG", "-shared"]
            cmd += dali_stub_src.split()
            cmd += ["-o"]
            cmd += dali_stub_lib.split()
            cmd += ["-fPIC"]
            cmd += dali_cflags.split()
            cmd += cuda_cflags.split()
            cmd += cuda_lflags.split()
            cmd += ["-O2"]

            cmd = list(filter(lambda x: len(x) != 0, cmd))
            print("Building DALI stub lib:\n\n " + " ".join(cmd) + "\n\n")
            subprocess.check_call(cmd, cwd=self.src_path, shell=False)  # nosec B603

            tf_cflags, tf_lflags = get_tf_build_flags()

            filenames = ["daliop.cc", "dali_dataset_op.cc"]
            plugin_src = ""
            for filename in filenames:
                plugin_src = plugin_src + " " + os.path.join(self.src_path, filename)

            lib_filename = "libdali_tf_current.so"
            lib_path = os.path.join(self.plugin_dest_dir, lib_filename)

            # for a newer TF we need to compiler with C++17
            cpp_ver = "--std=c++14" if Version(self.tf_version) < Version("2.10") else "--std=c++17"
            # Note: DNDEBUG flag is needed due to issue with TensorFlow custom ops:
            # https://github.com/tensorflow/tensorflow/issues/17316
            # Do not remove it.
            # the latest TF in conda needs to include /PREFIX/include
            root_include = "-I" + os.getenv("PREFIX", default="/usr") + "/include"
            cmd = [
                compiler,
                "-Wl,-R,$ORIGIN/..",
                "-Wl,-rpath,$ORIGIN",
                cpp_ver,
                "-DNDEBUG",
                "-shared",
            ]
            cmd += plugin_src.split()
            cmd += ["-o"]
            cmd += lib_path.split()
            cmd += ["-fPIC"]
            cmd += dali_cflags.split()
            cmd += tf_cflags.split()
            cmd += root_include.split()
            cmd += cuda_cflags.split()
            cmd += dali_lflags.split()
            cmd += tf_lflags.split()
            cmd += cuda_lflags.split()
            tf_versions = dict(
                zip(
                    ["TF_MAJOR_VERSION", "TF_MINOR_VERSION", "TF_PATCH_VERSION"],
                    self.tf_version.split("."),
                )
            )
            cmd += [f"-D{ver}={tf_versions[ver]}" for ver in tf_versions]
            cmd += ["-O2"]

            cmd = list(filter(lambda x: len(x) != 0, cmd))
            print("Build DALI TF library:\n\n " + " ".join(cmd) + "\n\n")
            subprocess.check_call(cmd, cwd=self.src_path, shell=False)  # nosec B603

            if not self.check_plugin(lib_path, dali_stub_lib):
                raise ImportError(
                    "Error while loading or testing the DALI TF plugin built "
                    + "from source, will not install"
                )
            print("Installation successful")

    def check_install_env(self):
        print("Checking build environment for DALI TF plugin ...")
        print(self.debug_str())

        if not self.tf_version or not self.tf_path:
            error_msg = "Installation error:"
            error_msg += (
                "\n Tensorflow installation not found. Install `tensorflow-gpu` " + "and try again"
            )
            error_msg += "\n" + self.debug_str()
            raise ImportError(error_msg)

        Path(self.plugin_dest_dir).mkdir(parents=True, exist_ok=True)

    def install(self):
        self.check_install_env()

        if self.prebuilt_exact_ver and self.install_prebuilt():
            return

        try:
            self.build()
        except Exception as e:
            print("Build from source failed with error: ", e)
            # If we haven't tried the prebuilt binary yet but there is one available, try now
            if self.can_install_prebuilt and not self.prebuilt_exact_ver:
                print("Trying to install prebuilt plugin")
                if self.install_prebuilt():
                    return
            raise e


def main():
    env = InstallerHelper()
    env.install()


if __name__ == "__main__":
    main()
