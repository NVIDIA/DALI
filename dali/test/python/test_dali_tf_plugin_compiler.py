# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import contextlib
import io
import os
import subprocess  # nosec B404
import unittest
from unittest.mock import patch

import dali_tf_plugin_utils as utils
import dali_tf_plugin_install_tool as installer


class CompilerTest(unittest.TestCase):
    def test_libcpp_flags(self):
        for flags, expected in (
            ("", False),
            ("-std=c++17 -stdlib=libstdc++", False),
            ("-std=c++20 -stdlib=libc++", True),
            ("-DVALUE=-stdlib=libc++", False),
        ):
            with self.subTest(flags=flags):
                self.assertEqual(utils.tf_uses_libcpp(flags), expected)

    def test_selection(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(utils, "which") as which:
            self.assertEqual(utils.get_cpp_compiler(), "g++")
            which.assert_not_called()
            which.return_value = b"/usr/bin/clang++"
            self.assertEqual(utils.get_cpp_compiler("-stdlib=libc++"), "clang++")
            which.return_value = None
            with self.assertRaisesRegex(ImportError, "Install a compatible Clang toolchain"):
                utils.get_cpp_compiler("-stdlib=libc++")

    def test_cxx_override(self):
        for compiler in ("/opt/clang++-22", "ccache g++", "g++ -m64"):
            with self.subTest(compiler=compiler), patch.dict(os.environ, {"CXX": compiler}):
                with patch.object(utils, "which") as which:
                    self.assertEqual(utils.get_cpp_compiler(), compiler)
                    self.assertEqual(utils.get_cpp_compiler("-stdlib=libc++"), compiler)
                    which.assert_not_called()

    def test_version_banners(self):
        for banner, expected in (
            ("g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0", "11.4"),
            ("Ubuntu clang version 18.1.3 (1ubuntu1)", "18.1"),
            ("clang version 22.1.0\nTarget: x86_64-unknown-linux-gnu", "22.1"),
        ):
            with self.subTest(banner=banner):
                with patch.object(utils.subprocess, "check_output", return_value=banner):
                    self.assertEqual(utils.get_cpp_compiler_version("c++"), expected)

    def test_invalid_version_banner(self):
        for banner in ("", "unrecognized compiler"):
            with self.subTest(banner=banner):
                with patch.object(utils.subprocess, "check_output", return_value=banner):
                    with self.assertRaisesRegex(ImportError, "Set CXX"):
                        utils.get_cpp_compiler_version("c++")


class InstallerCompilerTest(unittest.TestCase):
    def make_helper(
        self, compiler_error, prebuilt_version="2.23", tf_cflags="-std=c++20 -stdlib=libc++"
    ):
        with contextlib.ExitStack() as stack:
            overrides = {
                "get_module_path": "/installed",
                "get_tf_version": "2.23",
                "get_tf_build_flags": (tf_cflags, "-ltensorflow_framework"),
                "get_tf_compiler_version": "11.4",
                "which": None,
                "find_available_prebuilt_tf": prebuilt_version,
            }
            for name, value in overrides.items():
                stack.enter_context(patch.object(installer, name, return_value=value))
            stack.enter_context(patch.dict(os.environ, {"DALI_TF_ALWAYS_BUILD": "0"}))
            stack.enter_context(patch.object(installer.platform, "system", return_value="Linux"))
            stack.enter_context(patch.object(installer.platform, "machine", return_value="x86_64"))
            stack.enter_context(
                patch.object(installer, "find", side_effect=[["libdali.so"], ["plugin.so"]])
            )
            if compiler_error is None:
                stack.enter_context(
                    patch.object(installer, "get_cpp_compiler", return_value="clang++")
                )
                stack.enter_context(
                    patch.object(installer, "get_cpp_compiler_version", return_value="22.1")
                )
            elif isinstance(compiler_error, ImportError):
                stack.enter_context(
                    patch.object(installer, "get_cpp_compiler", side_effect=compiler_error)
                )
            else:
                stack.enter_context(patch.object(installer, "get_cpp_compiler", return_value="c++"))
                stack.enter_context(
                    patch.object(installer, "get_cpp_compiler_version", side_effect=compiler_error)
                )
            return installer.InstallerHelper()

    def test_prebuilt_without_compiler(self):
        for error in (
            ImportError("clang++ was not found"),
            FileNotFoundError("c++ was not found"),
            subprocess.CalledProcessError(1, ["c++", "--version"]),
        ):
            for version in ("2.23", "2.22"):
                with self.subTest(error=error, version=version):
                    helper = self.make_helper(error, version, tf_cflags="-std=c++17")
                    self.assertTrue(helper.can_install_prebuilt)
                    with patch.object(helper, "check_install_env"):
                        with patch.object(helper, "check_plugin", return_value=True):
                            with patch.object(installer, "copyfile") as copy:
                                with contextlib.redirect_stdout(io.StringIO()):
                                    helper.install()
                                copy.assert_called_once()

    def test_libcpp_skips_prebuilt(self):
        for version in ("2.23", "2.22"):
            for error in (None, ImportError("clang++ was not found")):
                with self.subTest(version=version, error=error):
                    helper = self.make_helper(error, version)
                    self.assertFalse(helper.can_install_prebuilt)
                    self.assertFalse(helper.prebuilt_exact_ver)
                    ok = subprocess.CompletedProcess([], 0, stdout="", stderr="")
                    with contextlib.ExitStack() as stack:
                        stack.enter_context(patch.object(helper, "check_install_env"))
                        prebuilt = stack.enter_context(patch.object(helper, "install_prebuilt"))
                        build = stack.enter_context(
                            patch.object(helper, "build", side_effect=helper.get_compiler)
                        )
                        stack.enter_context(
                            patch.object(installer.subprocess, "run", return_value=ok)
                        )
                        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                        if error is None:
                            helper.install()
                        else:
                            with self.assertRaisesRegex(ImportError, "clang\\+\\+ was not found"):
                                helper.install()
                        build.assert_called_once()
                        prebuilt.assert_not_called()

    def test_source_build_reports_environment(self):
        helper = self.make_helper(ImportError("clang++ was not found"))
        self.assertFalse(helper.can_default_compile)
        with self.assertRaisesRegex(ImportError, r"(?s)clang\+\+ was not found.*Environment:"):
            helper.get_compiler()

    def test_libcpp_probe(self):
        helper = self.make_helper(None)
        self.assertTrue(helper.tf_uses_libcpp)
        ok = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        with patch.object(installer.subprocess, "run", return_value=ok) as run:
            self.assertEqual(helper.get_compiler(), "clang++")
        cmd = run.call_args.args[0]
        self.assertEqual(cmd[0], "clang++")
        for flag in helper.tf_cflags.split():
            self.assertIn(flag, cmd)

        failed = subprocess.CompletedProcess([], 1, stdout="", stderr="cstdint not found")
        with patch.object(installer.subprocess, "run", return_value=failed):
            with self.assertRaises(ImportError) as ctx:
                helper.get_compiler()
        message = str(ctx.exception)
        self.assertIn("clang++", message)
        self.assertIn(helper.tf_cflags, message)
        self.assertIn("cstdint not found", message)

    def test_advice_not_repeated(self):
        helper = self.make_helper(ImportError("clang++ was not found, set CXX"))
        with self.assertRaises(ImportError) as ctx:
            helper.get_compiler()
        self.assertNotIn("Install a compatible C++ compiler", str(ctx.exception))
        helper = self.make_helper(FileNotFoundError("c++ was not found"))
        with self.assertRaisesRegex(ImportError, "Install a compatible C\\+\\+ compiler"):
            helper.get_compiler()


if __name__ == "__main__":
    unittest.main()
