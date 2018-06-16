// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
