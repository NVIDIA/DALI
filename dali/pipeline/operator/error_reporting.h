// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATOR_ERROR_REPORTING_H_
#define DALI_PIPELINE_OPERATOR_ERROR_REPORTING_H_

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "dali/core/api_helper.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/operator/name_utils.h"

namespace dali {

// TODO(klecki): Throw this one into a namespace?

/**
 * @brief Direct equivalent of Python's traceback.FrameSummary:
 * https://docs.python.org/3/library/traceback.html#traceback.FrameSummary
 * Describes a stack frame in the Python stack trace.
 */
struct PythonStackFrame {
  PythonStackFrame(std::string filename, int lineno, std::string name, std::string line)
      : filename(std::move(filename)),
        lineno(lineno),
        name(std::move(name)),
        line(std::move(line)) {}
  /** @brief File name of the source code executed for this frame. */
  std::string filename;
  /** @brief The line number of the source code for this frame. */
  int lineno;
  /** @brief Name of the function being executed in this frame. */
  std::string name;
  /** @brief A string representing the source code for this frame, with leading and trailing
   * whitespace stripped. */
  std::string line;
};

/**
 * @brief Get the origin stack trace for operator constructed with given spec.
 * The stack trace defines frames between invocation of the pipeline definition and the operator
 * call. The returned PythonStackFrame corresponds to Python traceback.FrameSummary, but the `line`
 * context may be invalid in some autograph transformed code.
 */
DLL_PUBLIC std::vector<PythonStackFrame> GetOperatorOriginInfo(const OpSpec &spec);

DLL_PUBLIC std::string FormatStack(const std::vector<PythonStackFrame> &stack_summary,
                                   bool include_context);

/**
 * @brief Simple structure capturing the exception to be propagated to the user with additional
 * context information (location of the error) and message.
 */
struct ErrorInfo {
  std::exception_ptr exception;
  std::string context_info;
  std::string additional_message;
};

[[noreturn]] void PropagateError(ErrorInfo error);


/**
 * @brief Base error class for exceptions produced in the DALI pipeline. Subtypes should be used
 * for specific errors that will be mapped into appropriate Python error types of matching names.
 *
 * DALI executor automatically extends those kind of errors when they are thrown from operators
 * and the origin information is present.
 */
class DaliError : public std::exception {
 public:
  // TODO(klecki): Do we want to mark when we have the origin info or introduce placeholders
  // for op name, etc?
  explicit DaliError(const std::string &msg) : std::exception(), msg_(msg) {}

  void UpdateMessage(const std::string &msg) {
    msg_ = msg;
  }

  const char *what() const noexcept override {
    return msg_.c_str();
  }

 private:
  std::string msg_;
};

/*
 Out of Python Error types: https://docs.python.org/3/library/exceptions.html#concrete-exceptions
 we skip (not applicable to DALI C++):
 * AttributeError - DALI doesn't implement dynamic obj.attr lookups that may error in the backend
 * EOFError - end of input() and raw_input()
 * FloatingPointError - not used in Python
 * GeneratorExit - not an error, a generator/coroutine.close() signal
 * ImportError, ModuleNotFoundError - Python module import related error
 * KeyError - error in dictionary lookups
 * KeyboardInterrupt - user triggered interrupt
 * MemoryError - recoverable memory errors
 * NameError, UnboundLocalError - lookup of unqualified names, Python code-level error
 * NotImplementedError - used for abstract base
 * OverflowError - we assume the code is safe :)
 * RecursionError
 * ReferenceError - Python weakref related errors
 * StopAsyncIteration - async iteration not supported in DALI
 * SyntaxError, IndentationError, TabError, SystemError - code and interpreter errors
 * ZeroDivisionError

 Errors considered for support (TODO(klecki)):
 * AssertionError - do we want to support actual Python-compatible asserts?
 * OSError -  consider mapping file system errors to Python vocabulary
 * SystemExit - may be useful for cleanup purposes
 * UnicodeError - if we ever do string processing?

 Errors currently supported:
 * RuntimeError
 * IndexError
 * ValueError
 * TypeError
 * StopIteration
 */

/**
 * @brief Error class that will be mapped to Python RuntimeError
 */
class DaliRuntimeError : public DaliError {
 public:
  explicit DaliRuntimeError(const std::string &msg) : DaliError(msg) {}
};

/**
 * @brief Error class that will be mapped to Python IndexError
 */
class DaliIndexError : public DaliError {
 public:
  explicit DaliIndexError(const std::string &msg) : DaliError(msg) {}
};

/**
 * @brief Error class that will be mapped to Python ValueError
 */
class DaliValueError : public DaliError {
 public:
  explicit DaliValueError(const std::string &msg) : DaliError(msg) {}
};

/**
 * @brief Error class that will be mapped to Python TypeError
 */
class DaliTypeError : public DaliError {
 public:
  explicit DaliTypeError(const std::string &msg) : DaliError(msg) {}
};

/**
 * @brief Error class that will be mapped to Python StopIteration
 */
class DaliStopIteration : public DaliError {
 public:
  explicit DaliStopIteration(const std::string &msg) : DaliError(msg) {}
};

/**
 * @brief Produce the error context message for given operator.
 * It contains:
 * * the name of the offending operator in the api variant it was instantiated from,
 * * the origin stack trace of the operator within pipeline definition.
 *
 * It can be prepended to the original error message.
 * @param message_name Will be used as the prefix of the error message, for example:
 * "Error in <device> operator <op_name>" or "Warning in <device> operator <op_name>"
 */
std::string GetErrorContextMessage(const OpSpec &spec, std::string_view message_name = "Error");

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_ERROR_REPORTING_H_
