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

import os
import subprocess
import unittest
from unittest.mock import patch

from dali_tf_plugin_install_tool import InstallerHelper
from dali_tf_plugin_utils import get_cpp_compiler, get_cpp_compiler_version


class CompilerSelectionTest(unittest.TestCase):
    def test_default_gcc(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_cpp_compiler("--std=c++17"), "g++")

    def test_libcpp_selects_clang(self):
        with patch.dict(os.environ, {}, clear=True), patch(
            "dali_tf_plugin_utils.which", return_value=b"/usr/bin/clang++"
        ):
            self.assertEqual(get_cpp_compiler("-stdlib=libc++ --std=c++20"), "clang++")

    def test_missing_clang(self):
        with patch.dict(os.environ, {}, clear=True), patch(
            "dali_tf_plugin_utils.which", return_value=None
        ):
            with self.assertRaisesRegex(ImportError, "clang\\+\\+ was not found"):
                get_cpp_compiler("-stdlib=libc++")

    def test_explicit_compiler(self):
        with patch.dict(os.environ, {"CXX": "/opt/llvm/bin/clang++"}, clear=True):
            self.assertEqual(get_cpp_compiler("-stdlib=libc++"), "/opt/llvm/bin/clang++")
            self.assertEqual(get_cpp_compiler(), "/opt/llvm/bin/clang++")

    def test_compiler_versions(self):
        banners = [
            ("g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0", "11.4"),
            ("g++ (GCC) 14.2.1 20250110", "14.2"),
            ("Ubuntu clang version 21.0.0 (build)", "21.0"),
            ("clang version 22.1.0 (https://github.com/llvm/llvm-project)", "22.1"),
            ("Apple clang version 17.0.0 (clang-1700.0.13.3)", "17.0"),
        ]
        for banner, version in banners:
            with self.subTest(banner=banner), patch(
                "dali_tf_plugin_utils.subprocess.check_output", return_value=banner
            ):
                self.assertEqual(get_cpp_compiler_version("c++"), version)

    def test_unknown_compiler_version(self):
        with patch("dali_tf_plugin_utils.subprocess.check_output", return_value="unknown"):
            with self.assertRaisesRegex(ImportError, "Could not determine"):
                get_cpp_compiler_version("c++")


class CompilerProbeTest(unittest.TestCase):
    def setUp(self):
        self.installer = InstallerHelper.__new__(InstallerHelper)
        self.installer.cpp_compiler = "clang++"
        self.installer.tf_uses_libcpp = True

    def test_libcpp_probe(self):
        with patch(
            "dali_tf_plugin_install_tool.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, "", ""),
        ) as run:
            self.assertEqual(self.installer.get_compiler(), "clang++")
            self.assertIn("-stdlib=libc++", run.call_args.args[0])
            self.assertIn("-std=c++20", run.call_args.args[0])

    def test_missing_libcpp_development_files(self):
        with patch(
            "dali_tf_plugin_install_tool.subprocess.run",
            return_value=subprocess.CompletedProcess([], 1, "", "'string' file not found"),
        ):
            with self.assertRaisesRegex(ImportError, "libc\\+\\+/libc\\+\\+abi development"):
                self.installer.get_compiler()

    def test_gcc_path_does_not_probe(self):
        self.installer.cpp_compiler = "g++"
        self.installer.tf_uses_libcpp = False
        self.installer.can_default_compile = True
        with patch("dali_tf_plugin_install_tool.subprocess.run") as run:
            self.assertEqual(self.installer.get_compiler(), "g++")
            run.assert_not_called()
