// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_PYTHON_PYTHON3_COMPAT_H_
#define DALI_PYTHON_PYTHON3_COMPAT_H_
#include <Python.h>

#if PY_MAJOR_VERSION >= 3
#define PyStr_Check PyUnicode_Check
#define PyStr_AsString PyUnicode_AsUTF8
#define PyInt_Check PyLong_Check
#define PyInt_AsLong PyLong_AsLong
#else
#define PyStr_Check PyString_Check
#define PyStr_AsString PyString_AsString
#endif

#endif  // DALI_PYTHON_PYTHON3_COMPAT_H_
