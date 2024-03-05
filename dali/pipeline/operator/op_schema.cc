// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#include <string>

#include "dali/core/error_handling.h"
#include "dali/core/python_util.h"
#include "dali/pipeline/operator/op_schema.h"
#include "dali/pipeline/operator/op_spec.h"

namespace dali {

std::map<string, OpSchema> &SchemaRegistry::registry() {
  static std::map<string, OpSchema> schema_map;
  return schema_map;
}

OpSchema &SchemaRegistry::RegisterSchema(const std::string &name) {
  auto &schema_map = registry();
  DALI_ENFORCE(schema_map.count(name) == 0,
               "OpSchema already "
               "registered for operator '" +
                   name +
                   "'. DALI_SCHEMA(op) "
                   "should only be called once per op.");

  // Insert the op schema and return a reference to it
  schema_map.emplace(std::make_pair(name, OpSchema(name)));
  return schema_map.at(name);
}

const OpSchema &SchemaRegistry::GetSchema(const std::string &name) {
  auto &schema_map = registry();
  auto it = schema_map.find(name);
  DALI_ENFORCE(it != schema_map.end(), "Schema for operator '" + name + "' not registered");
  return it->second;
}

const OpSchema *SchemaRegistry::TryGetSchema(const std::string &name) {
  auto &schema_map = registry();
  auto it = schema_map.find(name);
  return it != schema_map.end() ? &it->second : nullptr;
}


OpSchema::OpSchema(const std::string &name) : name_(name) {
  // Process the module path and operator name
  InitNames();
  // Fill internal arguments
  AddInternalArg("num_threads", "Number of CPU threads in a thread pool", -1);
  AddInternalArg("max_batch_size", "Max batch size", -1);
  AddInternalArg("device", "Device on which the Op is run", std::string("cpu"));
  AddInternalArg("inplace", "Whether Op can be run in place", false);
  AddInternalArg("default_cuda_stream_priority", "Default cuda stream priority", 0);
  AddInternalArg("checkpointing", "Setting to `true` enables checkpointing", false);

  AddOptionalArg("seed", R"code(Random seed.

If not provided, it will be populated based on the global seed of the pipeline.)code",
                 -1);

  AddOptionalArg("bytes_per_sample_hint", R"code(Output size hint, in bytes per sample.

If specified, the operator's outputs residing in GPU or page-locked host memory will be preallocated
to accommodate a batch of samples of this size.)code",
                 std::vector<int>{0});

  AddOptionalArg("preserve", R"code(Prevents the operator from being removed from the
graph even if its outputs are not used.)code",
                 false);

  // For simplicity we pass StackSummary as 4 separate arguments so we don't need to extend DALI
  // with support for special FrameSummary type.
  // List of FrameSummaries can be reconstructed using utility functions.
  AddOptionalArg("_origin_stack_filename", R"code(Every operator defined in Python captures and
processes the StackSummary (a List[FrameSummary], defined in Python traceback module) that describes
the callstack between the start of pipeline definition tracing and the "call" to the operator
(or full trace if the operator is defined outside the pipeline).
This information is propagated to the backend, so it can be later used to produce meaningful error
messages, pointing to the origin of the error in pipeline definition.

The list of FrameSummaries is split into four parameters: each is the list containing corresponding
parameters of FrameSummary. This parameter represents the `filename` member.)code",
                 std::vector<std::string>{});

  AddOptionalArg("_origin_stack_lineno", R"code(StackSummary - lineno member of FrameSummary, see
_origin_stack_filename for more information.)code",
                 std::vector<int>{});

  AddOptionalArg("_origin_stack_name", R"code(StackSummary - name member of FrameSummary, see
_origin_stack_filename for more information.)code",
                 std::vector<std::string>{});

  AddOptionalArg("_origin_stack_line", R"code(StackSummary - line member of FrameSummary, see
_origin_stack_filename for more information.)code",
                 std::vector<std::string>{});

  AddOptionalArg("_pipeline_internal", R"code(Boolean specifying if this operator was defined within
a pipeline scope. False if it was defined without pipeline being set as current.)code",
                 true);
  AddOptionalArg("_api",
                 "String identifying the Python API used to instantiate operator: "
                 "\"ops\" or \"fn\".",
                 "ops");

  AddOptionalArg("_display_name",
                 "Operator name as presented in the API it was instantiated in (without the module "
                 "path), for example: cast_like or CastLike.",
                 OperatorName());
}


const std::string &OpSchema::name() const {
  return name_;
}

const std::vector<std::string> &OpSchema::ModulePath() const {
  return module_path_;
}

const std::string &OpSchema::OperatorName() const {
  return operator_name_;
}

void OpSchema::InitNames() {
  static std::string kDelim = "__";
  std::string::size_type start_pos = 0;
  auto end_pos = name_.find(kDelim);
  while (end_pos != std::string::npos) {
    module_path_.push_back(name_.substr(start_pos, end_pos - start_pos));
    start_pos = end_pos + kDelim.size();
    end_pos = name_.find(kDelim, start_pos);
  }
  operator_name_ = name_.substr(start_pos);
}

OpSchema &OpSchema::DocStr(const string &dox) {
  dox_ = dox;
  return *this;
}


OpSchema &OpSchema::InputDox(int index, const string &name, const string &type_doc,
                             const string &doc) {
  CheckInputIndex(index);
  DALI_ENFORCE(!name.empty(), "Name of the argument should not be empty");
  DALI_ENFORCE(call_dox_.empty(),
               "Providing docstrings for inputs is not supported when the CallDocStr was used.");
  input_dox_set_ = true;
  input_dox_[index] = {name, type_doc, doc};
  return *this;
}


OpSchema &OpSchema::CallDocStr(const string &doc, bool append_kwargs_section) {
  DALI_ENFORCE(!doc.empty(), "The custom docstring for __call__ should not be empty.");

  DALI_ENFORCE(!input_dox_set_,
               "Providing docstring for `__call__` is not supported when docstrings for separate "
               "inputs were set using InputDox.");
  call_dox_ = doc;
  append_kwargs_section_ = append_kwargs_section;
  return *this;
}


OpSchema &OpSchema::OutputFn(SpecFunc f) {
  output_fn_ = std::move(f);
  return *this;
}


OpSchema &OpSchema::AdditionalOutputsFn(SpecFunc f) {
  additional_outputs_fn_ = std::move(f);
  return *this;
}


OpSchema &OpSchema::NumInput(int n) {
  DALI_ENFORCE(n >= 0);
  max_num_input_ = n;
  min_num_input_ = n;
  input_dox_.resize(n);
  input_layouts_.resize(n);
  input_devices_.resize(n);
  return *this;
}


OpSchema &OpSchema::NumInput(int min, int max) {
  DALI_ENFORCE(min <= max);
  DALI_ENFORCE(min >= 0);
  DALI_ENFORCE(max >= 0);
  min_num_input_ = min;
  max_num_input_ = max;
  input_layouts_.resize(max);
  input_dox_.resize(max);
  input_devices_.resize(max);
  return *this;
}


OpSchema &OpSchema::InputDevice(int first, int one_past, dali::InputDevice device) {
  for (int i = first; i < one_past; i++)
    input_devices_[i] = device;
  return *this;
}


OpSchema &OpSchema::InputDevice(int index, dali::InputDevice device) {
  input_devices_[index] = device;
  return *this;
}


DLL_PUBLIC dali::InputDevice OpSchema::GetInputDevice(int index) const {
  return input_devices_[index];
}


OpSchema &OpSchema::NumOutput(int n) {
  DALI_ENFORCE(n >= 0);
  num_output_ = n;
  return *this;
}


OpSchema &OpSchema::DisableAutoInputDox() {
  disable_auto_input_dox_ = true;
  return *this;
}


OpSchema &OpSchema::DisallowInstanceGrouping() {
  allow_instance_grouping_ = false;
  return *this;
}


OpSchema &OpSchema::SequenceOperator() {
  is_sequence_operator_ = true;
  return *this;
}


OpSchema &OpSchema::AllowSequences() {
  allow_sequences_ = true;
  return *this;
}


OpSchema &OpSchema::SupportVolumetric() {
  support_volumetric_ = true;
  return *this;
}


OpSchema &OpSchema::MakeInternal() {
  is_internal_ = true;
  return *this;
}


OpSchema &OpSchema::MakeDocHidden() {
  is_doc_hidden_ = true;
  return *this;
}


OpSchema &OpSchema::MakeDocPartiallyHidden() {
  is_doc_partially_hidden_ = true;
  return *this;
}


OpSchema &OpSchema::Deprecate(const std::string &in_favor_of, const std::string &explanation) {
  is_deprecated_ = true;
  deprecated_in_favor_of_ = in_favor_of;
  deprecation_message_ = explanation;
  return *this;
}


OpSchema &OpSchema::Unserializable() {
  serializable_ = false;
  return *this;
}


OpSchema &OpSchema::AddArg(const std::string &s, const std::string &doc, const DALIDataType dtype,
                           bool enable_tensor_input, bool support_per_frame_input) {
  CheckArgument(s);
  arguments_[s] = {doc, dtype};
  if (enable_tensor_input) {
    tensor_arguments_[s] = {support_per_frame_input};
  }
  return *this;
}


OpSchema &OpSchema::AddTypeArg(const std::string &s, const std::string &doc) {
  return AddArg(s, doc, DALI_DATA_TYPE);
}


OpSchema &OpSchema::InputLayout(int index, TensorLayout layout) {
  return InputLayout(index, {layout});
}


OpSchema &OpSchema::InputLayout(int index, std::initializer_list<TensorLayout> layouts) {
  CheckInputIndex(index);
  DALI_ENFORCE(input_layouts_[index].empty(),
               "Layouts for input " + std::to_string(index) + " already specified");
  for (auto &l : layouts) {
    DALI_ENFORCE(!l.empty(), "Cannot specify an empty layout for an input");
  }
  input_layouts_[index] = layouts;
  return *this;
}


OpSchema &OpSchema::InputLayout(TensorLayout layout) {
  return InputLayout({layout});
}


OpSchema &OpSchema::InputLayout(std::initializer_list<TensorLayout> layouts) {
  for (int i = 0; i < max_num_input_; i++)
    InputLayout(i, layouts);
  return *this;
}


const TensorLayout &OpSchema::GetInputLayout(int index, int sample_ndim,
                                             const TensorLayout &layout) const {
  CheckInputIndex(index);
  DALI_ENFORCE(layout.empty() || layout.ndim() == sample_ndim,
               make_string("The layout '", layout, "' is not valid for ", sample_ndim,
                           "-dimensional tensor"));

  if (input_layouts_[index].empty()) {
    return layout;
  }

  if (layout.empty()) {
    for (auto &l : input_layouts_[index])
      if (l.ndim() == sample_ndim)
        return l;
    std::stringstream ss;
    ss << "The number of dimensions " << sample_ndim
       << " does not match any of the allowed"
          " layouts for input "
       << index << ". Valid layouts are:\n";
    for (auto &l : input_layouts_[index])
      ss << l.c_str() << "\n";
    DALI_FAIL(ss.str());
  } else {
    for (auto &l : input_layouts_[index])
      if (l == layout)
        return l;
    std::stringstream ss;
    ss << "The layout \"" << layout
       << "\" does not match any of the allowed"
          " layouts for input "
       << index << ". Valid layouts are:\n";
    for (auto &l : input_layouts_[index])
      ss << l.c_str() << "\n";
    DALI_FAIL(ss.str());
  }
}

const std::vector<TensorLayout> &OpSchema::GetSupportedLayouts(int input_idx) const {
  CheckInputIndex(input_idx);
  return input_layouts_[input_idx];
}


OpSchema &OpSchema::AddOptionalArg(const std::string &s, const std::string &doc, DALIDataType dtype,
                                   std::nullptr_t, bool enable_tensor_input,
                                   bool support_per_frame_input) {
  CheckArgument(s);
  optional_arguments_[s] = {doc, dtype, nullptr, ShouldHideArgument(s)};
  if (enable_tensor_input) {
    tensor_arguments_[s] = {support_per_frame_input};
  }
  return *this;
}


OpSchema &OpSchema::AddOptionalTypeArg(const std::string &s, const std::string &doc,
                                       DALIDataType default_value) {
  CheckArgument(s);
  auto to_store = Value::construct(default_value);
  optional_arguments_[s] = {doc, DALI_DATA_TYPE, to_store.get(), ShouldHideArgument(s)};
  optional_arguments_unq_.push_back(std::move(to_store));
  return *this;
}


OpSchema &OpSchema::AddOptionalTypeArg(const std::string &s, const std::string &doc) {
  return AddOptionalArg<DALIDataType>(s, doc, nullptr);
}


OpSchema &OpSchema::AddOptionalArg(const std::string &s, const std::string &doc,
                                   const char *default_value) {
  return AddOptionalArg(s, doc, std::string(default_value), false);
}


OpSchema &OpSchema::DeprecateArgInFavorOf(const std::string &arg_name, std::string renamed_to,
                                          std::string msg) {
  if (msg.empty())
    msg = DefaultDeprecatedArgMsg(arg_name, renamed_to, false);
  deprecated_arguments_[arg_name] = {std::move(renamed_to), std::move(msg), false};
  return *this;
}


OpSchema &OpSchema::DeprecateArg(const std::string &arg_name, bool removed, std::string msg) {
  DALI_ENFORCE(
      HasArgument(arg_name),
      make_string("Argument \"", arg_name,
                  "\" has been marked for deprecation but it is not present in the schema."));
  if (msg.empty())
    msg = DefaultDeprecatedArgMsg(arg_name, {}, removed);
  deprecated_arguments_[arg_name] = {{}, std::move(msg), removed};
  return *this;
}


OpSchema &OpSchema::InPlaceFn(SpecFunc f) {
  (void)f;
  REPORT_FATAL_PROBLEM("In-place op support not yet implemented.");
  return *this;
}


OpSchema &OpSchema::AddParent(const std::string &parentName) {
  parents_.push_back(parentName);
  return *this;
}


OpSchema &OpSchema::NoPrune() {
  no_prune_ = true;
  return *this;
}


OpSchema &OpSchema::PassThrough(const std::map<int, int> &inout) {
  std::set<int> outputs;
  for (const auto &elems : inout) {
    outputs.insert(elems.second);
  }
  DALI_ENFORCE(inout.size() == outputs.size(),
               "Pass through can be defined only as 1-1 mapping between inputs and outputs, "
               "without duplicates.");
  DALI_ENFORCE(!HasSamplewisePassThrough(), "Two different modes of pass through can't be mixed.");
  passthrough_map_ = inout;
  return *this;
}


OpSchema &OpSchema::SamplewisePassThrough() {
  DALI_ENFORCE(!HasStrictPassThrough(), "Two different modes of pass through can't be mixed.");
  samplewise_any_passthrough_ = true;
  return *this;
}


const vector<std::string> &OpSchema::GetParents() const {
  return parents_;
}


string OpSchema::Dox() const {
  return dox_;
}


DLL_PUBLIC bool OpSchema::CanUseAutoInputDox() const {
  return !disable_auto_input_dox_ && MaxNumInput() <= 1;
}


DLL_PUBLIC bool OpSchema::AppendKwargsSection() const {
  return append_kwargs_section_;
}


DLL_PUBLIC bool OpSchema::HasCallDox() const {
  return !call_dox_.empty();
}


DLL_PUBLIC std::string OpSchema::GetCallDox() const {
  DALI_ENFORCE(HasCallDox(), "__call__ docstring was not set");
  return call_dox_;
}


DLL_PUBLIC bool OpSchema::HasInputDox() const {
  return input_dox_set_;
}


DLL_PUBLIC std::string OpSchema::GetCallSignatureInputs() const {
  DALI_ENFORCE(HasInputDox(), "Input documentation was not specified for this operator.");
  std::stringstream result;
  for (int i = 0; i < MinNumInput(); i++) {
    result << input_dox_[i].name;
    if (i < MaxNumInput() - 1) {
      result << ", ";
    }
  }
  for (int i = MinNumInput(); i < MaxNumInput(); i++) {
    result << input_dox_[i].name << " = None";
    if (i < MaxNumInput() - 1) {
      result << ", ";
    }
  }
  return result.str();
}


DLL_PUBLIC std::string OpSchema::GetInputName(int input_idx) const {
  CheckInputIndex(input_idx);
  DALI_ENFORCE(HasInputDox(), "Input documentation was not specified for this operator.");
  DALI_ENFORCE(!input_dox_[input_idx].name.empty(),
               make_string("Docstring for input ", input_idx,
                           "was not set. All inputs should be documented."));
  return input_dox_[input_idx].name;
}


DLL_PUBLIC std::string OpSchema::GetInputType(int input_idx) const {
  CheckInputIndex(input_idx);
  DALI_ENFORCE(HasInputDox(), "Input documentation was not specified for this operator.");
  return input_dox_[input_idx].type_doc;
}


DLL_PUBLIC std::string OpSchema::GetInputDox(int input_idx) const {
  CheckInputIndex(input_idx);
  DALI_ENFORCE(HasInputDox(), "Input documentation was not specified for this operator.");
  return input_dox_[input_idx].doc;
}


int OpSchema::MaxNumInput() const {
  return max_num_input_;
}


int OpSchema::MinNumInput() const {
  return min_num_input_;
}


int OpSchema::NumOutput() const {
  return num_output_;
}


bool OpSchema::AllowsInstanceGrouping() const {
  return allow_instance_grouping_;
}


bool OpSchema::IsSequenceOperator() const {
  return is_sequence_operator_;
}


bool OpSchema::AllowsSequences() const {
  return allow_sequences_;
}


bool OpSchema::SupportsVolumetric() const {
  return support_volumetric_;
}


bool OpSchema::IsInternal() const {
  return is_internal_;
}


bool OpSchema::IsDocHidden() const {
  return is_doc_hidden_;
}


bool OpSchema::IsDocPartiallyHidden() const {
  return is_doc_partially_hidden_;
}


bool OpSchema::IsDeprecated() const {
  return is_deprecated_;
}


const std::string &OpSchema::DeprecatedInFavorOf() const {
  return deprecated_in_favor_of_;
}


const std::string &OpSchema::DeprecationMessage() const {
  return deprecation_message_;
}


DLL_PUBLIC bool OpSchema::IsDeprecatedArg(const std::string &arg_name) const {
  if (deprecated_arguments_.find(arg_name) != deprecated_arguments_.end())
    return true;
  for (const auto &parent_name : parents_) {
    const OpSchema &parent = SchemaRegistry::GetSchema(parent_name);
    if (parent.IsDeprecatedArg(arg_name))
      return true;
  }
  return false;
}


DLL_PUBLIC const DeprecatedArgDef &OpSchema::DeprecatedArgMeta(const std::string &arg_name) const {
  auto it = deprecated_arguments_.find(arg_name);
  if (it != deprecated_arguments_.end()) {
    return it->second;
  }
  for (const auto &parent_name : parents_) {
    const OpSchema &parent = SchemaRegistry::GetSchema(parent_name);
    if (parent.IsDeprecatedArg(arg_name))
      return parent.DeprecatedArgMeta(arg_name);
  }
  DALI_FAIL(make_string("No deprecation metadata for argument \"", arg_name, "\" found."));
}


bool OpSchema::HasOutputFn() const {
  return static_cast<bool>(output_fn_) || static_cast<bool>(additional_outputs_fn_);
}


bool OpSchema::IsNoPrune() const {
  return no_prune_;
}


bool OpSchema::IsSerializable() const {
  return serializable_;
}


std::vector<int> OpSchema::GetPassThroughOutputIdx(int input_idx, const OpSpec &spec,
                                                   bool strict) const {
  if (samplewise_any_passthrough_) {
    // We indicate that we may pass through to any output
    int num_outputs = CalculateOutputs(spec) + CalculateAdditionalOutputs(spec);
    std::vector<int> result(num_outputs, 0);
    std::iota(result.begin(), result.end(), 0);
    return result;
  }
  auto it = passthrough_map_.find(input_idx);
  if (it == passthrough_map_.end())
    return {};
  return {it->second};
}


bool OpSchema::IsPassThrough(int input_idx, int output_idx, bool strict) const {
  if (!strict && HasSamplewisePassThrough()) {
    return true;
  }
  auto it = passthrough_map_.find(input_idx);
  if (it == passthrough_map_.end())
    return false;
  return it->second == output_idx;
}


bool OpSchema::HasPassThrough() const {
  return HasStrictPassThrough() || HasSamplewisePassThrough();
}


bool OpSchema::HasStrictPassThrough() const {
  return !passthrough_map_.empty();
}


bool OpSchema::HasSamplewisePassThrough() const {
  return samplewise_any_passthrough_;
}


int OpSchema::CalculateOutputs(const OpSpec &spec) const {
  if (!output_fn_) {
    return num_output_;
  } else {
    return output_fn_(spec);
  }
}


DLL_PUBLIC int OpSchema::CalculateAdditionalOutputs(const OpSpec &spec) const {
  if (!additional_outputs_fn_)
    return 0;
  return additional_outputs_fn_(spec);
}


bool OpSchema::SupportsInPlace(const OpSpec &spec) const {
  if (!in_place_fn_)
    return false;
  return in_place_fn_(spec);
}


void OpSchema::CheckArgs(const OpSpec &spec) const {
  std::vector<string> vec = spec.ListArguments();
  std::set<std::string> req_arguments_left;
  auto required_arguments = GetRequiredArguments();
  for (auto &arg_pair : required_arguments) {
    req_arguments_left.insert(arg_pair.first);
  }
  for (const auto &s : vec) {
    DALI_ENFORCE(HasArgument(s) || internal_arguments_.find(s) != internal_arguments_.end(),
                 "Got an unexpected argument \"" + s + "\"");
    std::set<std::string>::iterator it = req_arguments_left.find(s);
    if (it != req_arguments_left.end()) {
      req_arguments_left.erase(it);
    }
  }
  if (!req_arguments_left.empty()) {
    std::string ret = "Not all required arguments were specified for op \"" + this->name() +
                      "\". Please specify values for arguments: ";
    for (auto &str : req_arguments_left) {
      ret += "\"" + str + "\", ";
    }
    ret.erase(ret.size() - 2);
    ret += ".";
    DALI_FAIL(ret);
  }
}


bool OpSchema::HasRequiredArgument(const std::string &name, bool local_only) const {
  bool ret = arguments_.find(name) != arguments_.end();
  if (ret || local_only) {
    return ret;
  }
  for (const auto &p : parents_) {
    const OpSchema &parent = SchemaRegistry::GetSchema(p);
    ret = ret || parent.HasRequiredArgument(name);
  }
  return ret;
}


bool OpSchema::HasOptionalArgument(const std::string &name, bool local_only) const {
  bool ret = optional_arguments_.find(name) != optional_arguments_.end();
  if (ret || local_only) {
    return ret;
  }
  for (const auto &p : parents_) {
    const OpSchema &parent = SchemaRegistry::GetSchema(p);
    ret = ret || parent.HasOptionalArgument(name);
  }
  return ret;
}


bool OpSchema::HasInternalArgument(const std::string &name, bool local_only) const {
  bool ret = internal_arguments_.find(name) != internal_arguments_.end();
  if (ret || local_only) {
    return ret;
  }
  for (const auto &p : parents_) {
    const OpSchema &parent = SchemaRegistry::GetSchema(p);
    ret = ret || parent.HasInternalArgument(name);
  }
  return ret;
}


std::string OpSchema::GetArgumentDox(const std::string &name) const {
  DALI_ENFORCE(HasArgument(name),
               "Argument \"" + name + "\" is not supported by operator \"" + this->name() + "\".");
  if (HasRequiredArgument(name)) {
    return GetRequiredArguments().at(name).doc;
  } else {
    return GetOptionalArguments().at(name).doc;
  }
}


DALIDataType OpSchema::GetArgumentType(const std::string &name) const {
  DALI_ENFORCE(HasArgument(name),
               "Argument \"" + name + "\" is not supported by operator \"" + this->name() + "\".");
  if (HasRequiredArgument(name)) {
    return GetRequiredArguments().at(name).dtype;
  } else {
    return GetOptionalArguments().at(name).dtype;
  }
}


bool OpSchema::HasArgumentDefaultValue(const std::string &name) const {
  DALI_ENFORCE(HasArgument(name, true),
               "Argument \"" + name + "\" is not supported by operator \"" + this->name() + "\".");
  if (HasRequiredArgument(name)) {
    return false;
  }
  if (HasInternalArgument(name, true)) {
    return true;
  }
  auto *value_ptr = GetOptionalArguments().at(name).default_value;
  return value_ptr != nullptr;
}


std::string OpSchema::GetArgumentDefaultValueString(const std::string &name) const {
  DALI_ENFORCE(HasOptionalArgument(name), "Argument \"" + name +
                                              "\" is either not supported by operator \"" +
                                              this->name() + "\" or is not optional.");

  auto *value_ptr = GetOptionalArguments().at(name).default_value;

  DALI_ENFORCE(value_ptr,
               make_string("Argument \"", name,
                           "\" in operator \"" + this->name() + "\" has no default value."));

  auto &val = *value_ptr;
  auto str = val.ToString();
  if (val.GetTypeId() == DALI_STRING || val.GetTypeId() == DALI_TENSOR_LAYOUT) {
    return python_repr(str);
  } else if (val.GetTypeId() == DALI_STRING_VEC) {
    auto str_vec = dynamic_cast<ValueInst<std::vector<std::string>> &>(val).Get();
    return python_repr(str_vec);
  } else {
    return str;
  }
}


std::vector<std::string> OpSchema::GetArgumentNames() const {
  std::vector<std::string> ret;
  const auto &required = GetRequiredArguments();
  const auto &optional = GetOptionalArguments();
  const auto &deprecated = GetDeprecatedArguments();
  for (auto &arg_pair : required) {
    ret.push_back(arg_pair.first);
  }
  for (auto &arg_pair : optional) {
    if (!arg_pair.second.hidden) {
      ret.push_back(arg_pair.first);
    }
  }
  for (auto &arg_pair : deprecated) {
    // Deprecated aliases only appear in `deprecated` but regular
    // deprecated arguments appear both in `deprecated` and either `required` or `optional`.
    if (required.find(arg_pair.first) == required.end() &&
        optional.find(arg_pair.first) == optional.end())
      ret.push_back(arg_pair.first);
  }
  return ret;
}


bool OpSchema::IsTensorArgument(const std::string &name) const {
  return FindTensorArgument(name);
}


bool OpSchema::ArgSupportsPerFrameInput(const std::string &arg_name) const {
  auto arg_desc = FindTensorArgument(arg_name);
  return arg_desc && arg_desc->supports_per_frame;
}


const TensorArgDesc *OpSchema::FindTensorArgument(const std::string &name) const {
  auto it = tensor_arguments_.find(name);
  if (it != tensor_arguments_.end()) {
    return &it->second;
  }
  for (const auto &p : parents_) {
    const OpSchema &parent = SchemaRegistry::GetSchema(p);
    auto desc = parent.FindTensorArgument(name);
    if (desc) {
      return desc;
    }
  }
  return nullptr;
}


void OpSchema::CheckArgument(const std::string &s) {
  DALI_ENFORCE(!HasArgument(s, false, true), "Argument \"" + s + "\" already added to the schema");
  DALI_ENFORCE(internal_arguments_.find(s) == internal_arguments_.end(),
               "Argument name \"" + s + "\" is reserved for internal use");
}


void OpSchema::CheckInputIndex(int index) const {
  DALI_ENFORCE(index >= 0 && index < max_num_input_,
               "Output index (=" + std::to_string(index) + ") out of range [0.." +
                   std::to_string(max_num_input_) + ").\nWas NumInput called?");
}


std::string OpSchema::DefaultDeprecatedArgMsg(const std::string &arg_name,
                                              const std::string &renamed_to, bool removed) const {
  std::stringstream ss;
  if (removed) {
    ss << "The argument ``" << arg_name
       << "`` is no longer used and will be removed in a future release.";
  } else if (!renamed_to.empty()) {
    ss << "The argument ``" << arg_name << "`` is a deprecated alias for ``" << renamed_to
       << "``. Use ``" << renamed_to << "`` instead.";
  } else {
    ss << "The argument ``" << arg_name << "`` is now deprecated and its usage is discouraged.";
  }
  return ss.str();
}


std::map<std::string, RequiredArgumentDef> OpSchema::GetRequiredArguments() const {
  auto ret = arguments_;
  for (const auto &parent_name : parents_) {
    const OpSchema &parent = SchemaRegistry::GetSchema(parent_name);
    const auto &parent_args = parent.GetRequiredArguments();
    ret.insert(parent_args.begin(), parent_args.end());
  }
  return ret;
}


std::map<std::string, DefaultedArgumentDef> OpSchema::GetOptionalArguments() const {
  auto ret = optional_arguments_;
  for (const auto &parent_name : parents_) {
    const OpSchema &parent = SchemaRegistry::GetSchema(parent_name);
    const auto &parent_args = parent.GetOptionalArguments();
    ret.insert(parent_args.begin(), parent_args.end());
  }
  return ret;
}


std::map<std::string, DeprecatedArgDef> OpSchema::GetDeprecatedArguments() const {
  auto ret = deprecated_arguments_;
  for (const auto &parent_name : parents_) {
    const OpSchema &parent = SchemaRegistry::GetSchema(parent_name);
    const auto &parent_args = parent.GetDeprecatedArguments();
    ret.insert(parent_args.begin(), parent_args.end());
  }
  return ret;
}


std::pair<const OpSchema *, const Value *> OpSchema::FindDefaultValue(const std::string &name,
                                                                      bool local_only,
                                                                      bool include_internal) const {
  auto it = optional_arguments_.find(name);
  if (it != optional_arguments_.end()) {
    return {this, it->second.default_value};
  }
  if (include_internal) {
    it = internal_arguments_.find(name);
    if (it != internal_arguments_.end()) {
      return {this, it->second.default_value};
    }
  }
  if (local_only)
    return {nullptr, nullptr};

  for (const auto &p : parents_) {
    const OpSchema &parent = SchemaRegistry::GetSchema(p);
    auto schema_val = parent.FindDefaultValue(name, false, include_internal);
    if (schema_val.first && schema_val.second)
      return schema_val;
  }
  return {nullptr, nullptr};
}


bool OpSchema::HasArgument(const std::string &name,
                           bool include_internal,
                           bool local_only) const {
  return HasRequiredArgument(name, local_only) || HasOptionalArgument(name, local_only) ||
         (include_internal && HasInternalArgument(name, true));
}

}  // namespace dali
