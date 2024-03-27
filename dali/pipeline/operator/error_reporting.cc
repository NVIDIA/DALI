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

#include <fmt/format.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
// for fmt::join
#include <fmt/ranges.h>
// for ostream support
#include <fmt/ostream.h>

#include "dali/core/error_handling.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/error_reporting.h"
#include "dali/pipeline/operator/name_utils.h"
#include "dali/pipeline/operator/op_spec.h"

// template <> struct fmt::formatter<dali::DALIDataType> : fmt::ostream_formatter {};
// template <> struct fmt::formatter<dali::DALIDataType> :
// fmt::tostring_formatter<dali::DALIDataType> {};


namespace dali {

auto format_as(dali::DALIDataType type) {
  return dali::to_string(type);
}

std::vector<PythonStackFrame> GetOperatorOriginInfo(const OpSpec &spec) {
  auto origin_stack_filename = spec.GetRepeatedArgument<std::string>("_origin_stack_filename");
  auto origin_stack_lineno = spec.GetRepeatedArgument<int>("_origin_stack_lineno");
  auto origin_stack_name = spec.GetRepeatedArgument<std::string>("_origin_stack_name");
  auto origin_stack_line = spec.GetRepeatedArgument<std::string>("_origin_stack_line");

  std::vector<PythonStackFrame> origin_stack;
  origin_stack.reserve(origin_stack_filename.size());
  const char error[] = "Internal error, mismatch in origin stack trace data.";
  DALI_ENFORCE(origin_stack_filename.size() == origin_stack_lineno.size(), error);
  DALI_ENFORCE(origin_stack_filename.size() == origin_stack_name.size(), error);
  DALI_ENFORCE(origin_stack_filename.size() == origin_stack_line.size(), error);
  for (size_t i = 0; i < origin_stack_filename.size(); i++) {
    origin_stack.emplace_back(std::move(origin_stack_filename[i]), origin_stack_lineno[i],
                              std::move(origin_stack_name[i]), std::move(origin_stack_line[i]));
  }
  return origin_stack;
}


std::string FormatStack(const std::vector<PythonStackFrame> &stack_summary, bool include_context) {
  std::stringstream s;
  for (auto &frame_summary : stack_summary) {
    s << "  File \"" << frame_summary.filename << "\", line " << frame_summary.lineno << ", in "
      << frame_summary.name << "\n";
    // Python doesn't report empty lines
    if (include_context && frame_summary.line.size()) {
      s << "    " << frame_summary.line << "\n";
    }
  }
  return s.str();
}

void PropagateError(ErrorInfo error) {
  try {
    if (error.exception) {
      std::rethrow_exception(error.exception);
    }
  }
  // DALI <-> Python mapped type errors:
  catch (DaliError &e) {
    e.UpdateMessage(make_string(error.context_info, e.what(), error.additional_message));
    throw;
  } catch (DALIException &e) {
    // We drop the C++ stack trace at this point and go back to runtime_error.
    throw std::runtime_error(
        make_string(error.context_info, e.what(),
                    "\nC++ context: " + e.GetFileAndLine() + error.additional_message));
  }
  // Exceptions that are mapped by pybind from C++ into a sensible C++ one:
  catch (std::invalid_argument &e) {
    throw std::invalid_argument(
        make_string(error.context_info, e.what(), error.additional_message));
  } catch (std::domain_error &e) {
    throw std::domain_error(make_string(error.context_info, e.what(), error.additional_message));
  } catch (std::length_error &e) {
    throw std::length_error(make_string(error.context_info, e.what(), error.additional_message));
  } catch (std::out_of_range &e) {
    throw std::out_of_range(make_string(error.context_info, e.what(), error.additional_message));
  } catch (std::range_error &e) {
    throw std::range_error(make_string(error.context_info, e.what(), error.additional_message));
  }
  // Map the rest into runtime error (it would happen this way regardless)
  catch (std::exception &e) {
    throw std::runtime_error(make_string(error.context_info, e.what(), error.additional_message));
  } catch (...) {
    throw std::runtime_error(
        make_string(error.context_info, "Unknown critical error.", error.additional_message));
  }
}

std::string GetErrorContextMessage(const OpSpec &spec) {
  auto device = spec.GetArgument<std::string>("device");
  auto op_name = GetOpDisplayName(spec, true);
  std::transform(device.begin(), device.end(), device.begin(), ::toupper);

  auto origin_stack_trace = GetOperatorOriginInfo(spec);
  auto formatted_origin_stack = FormatStack(origin_stack_trace, true);
  auto optional_stack_mention =
      formatted_origin_stack.size() ?
          (",\nwhich was used in the pipeline definition with the following traceback:\n\n" +
           formatted_origin_stack + "\n") :
          " ";  // we need space before "encountered"

  return make_string("Error in ", device, " operator `", op_name, "`", optional_stack_mention,
                     "encountered:\n\n");
}


namespace validate {

std::string SepIfNotEmpty(const std::string &str, const std::string &sep = " ") {
  if (str.empty()) {
    return "";
  }
  return sep;
}

DALIDataType Type(DALIDataType actual_type, DALIDataType expected_type, const std::string &name,
                  const std::string &additional_msg) {
  if (actual_type == expected_type) {
    return actual_type;
  }

  throw DaliTypeError(fmt::format("Unexpected type for {}. Got type: `{}` but expected: `{}`.{}{}",
                                  name, actual_type, expected_type, SepIfNotEmpty(additional_msg),
                                  additional_msg));
}

DALIDataType Type(DALIDataType actual_type, span<const DALIDataType> expected_types,
                  const std::string &name, const std::string &additional_msg) {
  if (std::size(expected_types) == 1) {
    return Type(actual_type, expected_types[0], name, additional_msg);
  }
  for (auto expected_type : expected_types) {
    if (actual_type == expected_type) {
      return actual_type;
    }
  }

  throw DaliTypeError(fmt::format(
      "Unexpected type for {}. Got type: `{}` but expected one of: `{}`.{}{}", name, actual_type,
      fmt::join(expected_types, "`, `"), SepIfNotEmpty(additional_msg), additional_msg));
}

DALIDataType InputType(const OpSpec &spec, const Workspace &ws, int input_idx,
                       DALIDataType allowed_type, const std::string &additional_msg) {
  DALIDataType dtype = ws.GetInputDataType(input_idx);
  return Type(dtype, allowed_type, FormatInput(spec, input_idx), additional_msg);
}

DALIDataType InputType(const OpSpec &spec, const Workspace &ws, int input_idx,
                       span<const DALIDataType> allowed_types, const std::string &additional_msg) {
  DALIDataType dtype = ws.GetInputDataType(input_idx);
  return Type(dtype, allowed_types, FormatInput(spec, input_idx), additional_msg);
}

DALIDataType Dtype(const OpSpec &spec, DALIDataType allowed_type, bool allow_unspecified,
                   const std::string &additional_msg) {
  if (allow_unspecified && !spec.HasArgument("dtype")) {
    return DALI_NO_TYPE;
  } else if (!allow_unspecified && !spec.HasArgument("dtype")) {
    throw DaliValueError(fmt::format("{} was not specified.{}{}",
                                     FormatArgument(spec, "dtype", true),
                                     SepIfNotEmpty(additional_msg), additional_msg));
  }
  return Type(spec.GetArgument<DALIDataType>("dtype"), allowed_type, FormatArgument(spec, "dtype"),
              additional_msg);
}

DALIDataType Dtype(const OpSpec &spec, span<const DALIDataType> allowed_types,
                   bool allow_unspecified, const std::string &additional_msg) {
  if (allow_unspecified && !spec.HasArgument("dtype")) {
    return DALI_NO_TYPE;
  } else if (!allow_unspecified && !spec.HasArgument("dtype")) {
    throw DaliValueError(fmt::format("{} was not specified.{}{}",
                                     FormatArgument(spec, "dtype", true),
                                     SepIfNotEmpty(additional_msg), additional_msg));
  }
  return Type(spec.GetArgument<DALIDataType>("dtype"), allowed_types, FormatArgument(spec, "dtype"),
              additional_msg);
}

void Dim(int actual_dim, int expected_dim, const std::string &name,
         const std::string &additional_msg) {
  if (actual_dim == expected_dim) {
    return;
  }
  throw DaliValueError(fmt::format("Got dim: `{}` for {}, but expected: `{}`.{}{}", actual_dim,
                                   name, expected_dim, SepIfNotEmpty(additional_msg),
                                   additional_msg));
}

DALIDataType Dtype(const OpSpec &spec, const Workspace &ws, bool (*is_valid)(DALIDataType),
                   const std::string &explanation) {
  return DALI_NO_TYPE;  // TODO(klecki): implement
}

DALIDataType OutputType(const OpSpec &spec, const Workspace &ws, int output_idx,
                        DALIDataType allowed_type, const std::string &additional_msg) {
  DALIDataType dtype = ws.GetOutputDataType(output_idx);
  return Type(dtype, allowed_type, FormatOutput(spec, output_idx), additional_msg);
}

DALIDataType OutputType(const OpSpec &spec, const Workspace &ws, int output_idx,
                        span<const DALIDataType> allowed_types, const std::string &additional_msg) {
  DALIDataType dtype = ws.GetOutputDataType(output_idx);
  return Type(dtype, allowed_types, FormatOutput(spec, output_idx), additional_msg);
}

DALIDataType ArgumentType(const OpSpec &spec, const Workspace &ws, const std::string &arg_name,
                          const std::string &additional_msg) {
  DALIDataType expected_type = spec.GetSchema().GetArgumentType(arg_name);
  if (!spec.HasTensorArgument(arg_name)) {
    return expected_type;
  }
  return Type(ws.ArgumentInput(arg_name).type(), expected_type, FormatArgument(spec, arg_name),
              additional_msg);
}

}  // namespace validate
}  // namespace dali
