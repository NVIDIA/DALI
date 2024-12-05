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

std::map<string, OpSchema, std::less<>> &SchemaRegistry::registry() {
  static std::map<string, OpSchema, std::less<>> schema_map;
  return schema_map;
}

OpSchema &SchemaRegistry::RegisterSchema(std::string_view name) {
  auto &schema_map = registry();
  DALI_ENFORCE(schema_map.count(name) == 0,
               "OpSchema already "
               "registered for operator '" +
                   std::string(name) +
                   "'. DALI_SCHEMA(op) "
                   "should only be called once per op.");

  // Insert the op schema and return a reference to it
  auto [it, inserted] = schema_map.emplace(std::make_pair(name, OpSchema(name)));
  return it->second;
}

const OpSchema &SchemaRegistry::GetSchema(std::string_view name) {
  auto &schema_map = registry();
  auto it = schema_map.find(name);
  if (it == schema_map.end())
    throw std::out_of_range("Schema for operator '" + std::string(name) + "' not registered");

  return it->second;
}

const OpSchema *SchemaRegistry::TryGetSchema(std::string_view name) {
  auto &schema_map = registry();
  auto it = schema_map.find(name);
  return it != schema_map.end() ? &it->second : nullptr;
}

const OpSchema &OpSchema::Default() {
  static OpSchema default_schema("");
  return default_schema;
}

OpSchema::OpSchema(std::string_view name) : name_(name) {
  // Process the module path and operator name
  InitNames();
  // Fill internal arguments
  AddInternalArg("num_threads", "Number of CPU threads in a thread pool", -1);
  AddInternalArg("max_batch_size", "Max batch size", -1);
  AddInternalArg("device", "Device on which the Op is run", std::string("cpu"));
  AddInternalArg("inplace", "Whether Op can be run in place", false);
  AddInternalArg("default_cuda_stream_priority", "Default cuda stream priority", 0);  // deprecated
  AddInternalArg("checkpointing", "Setting to `true` enables checkpointing", false);

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

  std::string default_module = "nvidia.dali.ops";
  for (const auto &submodule : ModulePath()) {
    default_module += "." + submodule;
  }

  AddOptionalArg("_module",
                 "String identifying the module in which the operator is defined. "
                 "Most of the time it is `__module__` of the API function/class.",
                 default_module);

  AddOptionalArg("_display_name",
                 "Operator name as presented in the API it was instantiated in (without the module "
                 "path), for example: cast_like or CastLike.",
                 OperatorName());

  AddOptionalArg("seed",
                 "Random seed; if not set, one will be assigned automatically, if needed.",
                 -1_u64);
}


std::string_view OpSchema::name() const {
  return name_;
}

const std::vector<std::string> &OpSchema::ModulePath() const {
  return module_path_;
}

std::string_view OpSchema::OperatorName() const {
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

OpSchema &OpSchema::DocStr(std::string dox) {
  dox_ = std::move(dox);
  return *this;
}


OpSchema &OpSchema::InputDox(int index, std::string_view name, std::string type_doc,
                             std::string doc) {
  CheckInputIndex(index);
  DALI_ENFORCE(!name.empty(), "Name of the argument should not be empty");
  DALI_ENFORCE(call_dox_.empty(),
               "Providing docstrings for inputs is not supported when the CallDocStr was used.");
  input_dox_set_ = true;
  input_info_[index].name = name;
  input_info_[index].doc = {std::move(type_doc), std::move(doc)};
  return *this;
}


OpSchema &OpSchema::CallDocStr(std::string doc, bool append_kwargs_section) {
  DALI_ENFORCE(!doc.empty(), "The custom docstring for __call__ should not be empty.");

  DALI_ENFORCE(!input_dox_set_,
               "Providing docstring for `__call__` is not supported when docstrings for separate "
               "inputs were set using InputDox.");
  call_dox_ = std::move(doc);
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
  input_info_.resize(n);
  return *this;
}


OpSchema &OpSchema::NumInput(int min, int max) {
  DALI_ENFORCE(min <= max);
  DALI_ENFORCE(min >= 0);
  DALI_ENFORCE(max >= 0);
  min_num_input_ = min;
  max_num_input_ = max;
  input_info_.resize(max);
  return *this;
}


OpSchema &OpSchema::InputDevice(int first, int one_past, dali::InputDevice device) {
  for (int i = first; i < one_past; i++)
    input_info_[i].device = device;
  return *this;
}


OpSchema &OpSchema::InputDevice(int index, dali::InputDevice device) {
  input_info_[index].device = device;
  return *this;
}


DLL_PUBLIC dali::InputDevice OpSchema::GetInputDevice(int index) const {
  return input_info_[index].device;
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


OpSchema &OpSchema::Deprecate(std::string_view in_favor_of, std::string_view explanation) {
  is_deprecated_ = true;
  deprecated_in_favor_of_ = in_favor_of;
  deprecation_message_ = explanation;
  return *this;
}


OpSchema &OpSchema::Unserializable() {
  serializable_ = false;
  return *this;
}

ArgumentDef &OpSchema::AddArgumentImpl(std::string_view name,
                                       DALIDataType type,
                                       std::unique_ptr<Value> default_value,
                                       std::string doc) {
  if (Default().HasInternalArgument(name))
    throw std::invalid_argument(make_string(
      "The argument name `", name, "` is reserved for internal use"));

  auto [it, inserted] = arguments_.emplace(name);
  if (!inserted) {
    throw std::invalid_argument(make_string(
      "The schema for operator `", name_, `" already contains an argument `", name, "`."));
  }

  auto &arg = it->second;
  arg.defined_in = this;
  arg.dtype = type;
  arg.default_value = std::move(default_value);
  arg.doc = std::move(doc);

  if (ShouldHideArgument(s))
    arg.hidden = true;

  return arg;
}

OpSchema &OpSchema::AddArg(std::string_view s, std::string doc, const DALIDataType dtype,
                           bool enable_tensor_input, bool support_per_frame_input) {
  auto &arg = AddArgumentImpl(s, dtype, nullptr, std::move(doc));
  arg.required = true;
  arg.tensor = enable_tensor_input;
  if (arg.tensor)
    arg.per_frame = support_per_frame_input;
  return *this;
}


OpSchema &OpSchema::AddTypeArg(std::string_view s, std::string doc) {
  return AddArg(s, std::move(doc), DALI_DATA_TYPE);
}


OpSchema &OpSchema::InputLayout(int index, TensorLayout layout) {
  return InputLayout(index, {layout});
}


OpSchema &OpSchema::InputLayout(int index, std::initializer_list<TensorLayout> layouts) {
  CheckInputIndex(index);
  DALI_ENFORCE(input_info_[index].layouts.empty(),
               "Layouts for input " + std::to_string(index) + " already specified");
  for (auto &l : layouts) {
    DALI_ENFORCE(!l.empty(), "Cannot specify an empty layout for an input");
  }
  input_info_[index].layouts = layouts;
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

  if (input_info_[index].layouts.empty()) {
    return layout;
  }

  if (layout.empty()) {
    for (auto &l : input_info_[index].layouts)
      if (l.ndim() == sample_ndim)
        return l;
    std::stringstream ss;
    ss << "The number of dimensions " << sample_ndim
       << " does not match any of the allowed"
          " layouts for input "
       << index << ". Valid layouts are:\n";
    for (auto &l : input_info_[index].layouts)
      ss << l.c_str() << "\n";
    DALI_FAIL(ss.str());
  } else {
    for (auto &l : input_info_[index].layouts)
      if (l == layout)
        return l;
    std::stringstream ss;
    ss << "The layout \"" << layout
       << "\" does not match any of the allowed"
          " layouts for input "
       << index << ". Valid layouts are:\n";
    for (auto &l : input_info_[index].layouts)
      ss << l.c_str() << "\n";
    DALI_FAIL(ss.str());
  }
}

const std::vector<TensorLayout> &OpSchema::GetSupportedLayouts(int input_idx) const {
  CheckInputIndex(input_idx);
  return input_info_[input_idx].layouts;
}


OpSchema &OpSchema::AddOptionalArg(std::string_view s, std::string doc, DALIDataType dtype,
                                   std::nullptr_t, bool enable_tensor_input,
                                   bool support_per_frame_input) {
  auto &arg = AddArgumentImpl(s, dtype, nullptr, std::move(doc));
  arg.tensor = enable_tensor_input;
  if (arg.tensor)
    arg.per_frame = support_per_frame_input;
}


OpSchema &OpSchema::AddOptionalTypeArg(std::string_view name, std::string doc,
                                       DALIDataType default_value) {
  AddArgumentImpl(name, DALI_DATA_TYPE, Value::construct(default_value), std::move(doc));
  return *this;
}


OpSchema &OpSchema::AddOptionalTypeArg(std::string_view s, std::string doc) {
  return AddOptionalArg<DALIDataType>(s, std::move(doc), nullptr);
}


OpSchema &OpSchema::AddOptionalArg(std::string_view s, std::string doc,
                                   const char *default_value) {
  return AddOptionalArg(s, std::move(doc), std::string(default_value), false);
}

OpSchema &OpSchema::AddRandomSeedArg() {
  AddOptionalArg("seed",
                 "Random seed; if not set, one will be assigned automatically, if needed.",
                 -1_u64);
  return *this;
}

bool OpSchema::HasRandomSeedArg() const {
  return !IsDeprecatedArg("seed");
}

OpSchema &OpSchema::DeprecateArgInFavorOf(std::string_view arg_name, std::string renamed_to,
                                          std::string msg) {
  if (msg.empty())
    msg = DefaultDeprecatedArgMsg(arg_name, renamed_to, false);
  deprecated_arguments_[arg_name] = {std::move(renamed_to), std::move(msg), false};
  return *this;
}


OpSchema &OpSchema::DeprecateArg(std::string_view arg_name, bool removed, std::string msg) {
  DALI_ENFORCE(
      HasArgument(arg_name, true),
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


OpSchema &OpSchema::AddParent(std::string_view parentName) {
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


const vector<std::string> &OpSchema::GetParentNames() const {
  return parent_names_;
}

const vector<const OpSchema *> &OpSchema::GetParents() const {
  return parents_.Get([&]() {
    std::vector<const OpSchema *> parents;
    if (this == &Default())
      return parents;  // the default schema has no parents

    parents.reserve(parent_names_.size() + 1);  // add one more for the default
    for (auto &name : parent_names_) {
      parents.push_back(&SchemaRegistry::GetSchema(name));
    }
    parents.push_back(&Default());
    return parents;
  });
}

std::map<std::string, const ArgumentDef *, std::less<>> &OpSchema::GetFlattenedArguments() const {
  return flattened_arguments_.Get([&]() {
    std::map<std::string, ArgumentDef, std::less<>> args = arguments_;
    // This is slightly inefficient, because we'll go over the default arguments multiple times
    // but that's only once in schema's lifetime.
    for (auto *parent : GetParents()) {
      for (auto &[name, arg] : parent->GetFlattenedArguments())
        args.emplace(name, &arg);  // this will skip arguments defined in this schema
    }
    return args;
  });
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
    result << input_info_[i].name;
    if (i < MaxNumInput() - 1) {
      result << ", ";
    }
  }
  for (int i = MinNumInput(); i < MaxNumInput(); i++) {
    result << input_info_[i].name << " = None";
    if (i < MaxNumInput() - 1) {
      result << ", ";
    }
  }
  return result.str();
}


DLL_PUBLIC std::string OpSchema::GetInputName(int input_idx) const {
  CheckInputIndex(input_idx);
  DALI_ENFORCE(HasInputDox(), "Input documentation was not specified for this operator.");
  DALI_ENFORCE(!input_info_[input_idx].name.empty(),
               make_string("Docstring for input ", input_idx,
                           "was not set. All inputs should be documented."));
  return input_info_[input_idx].name;
}


DLL_PUBLIC std::string OpSchema::GetInputType(int input_idx) const {
  CheckInputIndex(input_idx);
  DALI_ENFORCE(HasInputDox(), "Input documentation was not specified for this operator.");
  return input_info_[input_idx].doc.type_doc;
}


DLL_PUBLIC std::string OpSchema::GetInputDox(int input_idx) const {
  CheckInputIndex(input_idx);
  DALI_ENFORCE(HasInputDox(), "Input documentation was not specified for this operator.");
  return input_info_[input_idx].doc.doc;
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


std::string_view OpSchema::DeprecatedInFavorOf() const {
  return deprecated_in_favor_of_;
}


std::string_view OpSchema::DeprecationMessage() const {
  return deprecation_message_;
}


DLL_PUBLIC bool OpSchema::IsDeprecatedArg(std::string_view arg_name) const {
  if (auto *arg = FindArgument(arg_name))

  return false;
}


DLL_PUBLIC const DeprecatedArgDef &OpSchema::DeprecatedArgInfo(std::string_view arg_name) const {
  auto &arg = GetArgument(arg_name);
  if (!arg.deprecated)
    DALI_FAIL(make_string("No deprecation info for argument \"", arg_name, "\" found."));
  return *arg.deprecated;
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

bool OpSchema::HasOptionalArgument(std::string_view name) const {
  return FindArgument(name, [](const ArgumentDef &arg) {
    return !arg.required;
  });
}

const ArgumentDef &OpSchema::GetArgument(std::string_view name) const {
  if (auto *arg = FindArgument(name))
    return *arg;
  throw std::out_of_range(make_string(
        "Argument \"", name, "\" is not supported by operator \"", this->name(), "\".");
}

std::string OpSchema::GetArgumentDox(std::string_view name) const {
  return GetArgument(name).doc;
}


DALIDataType OpSchema::GetArgumentType(std::string_view name) const {
  return GetArgument(name).dtype;

}


bool OpSchema::HasArgumentDefaultValue(std::string_view name) const {
  return GetArgument(name).default_value
}


std::string OpSchema::GetArgumentDefaultValueString(std::string_view name) const {
  auto *value_ptr = GetArgument(name).default_value.get();

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
  const auto &args = GetFlattenedArguments();
  for (auto it = args.begin(); it != args.end(); ++it)
    if (!it->second->hidden)
      ret.push_back(it->first);
  return ret;
}


bool OpSchema::IsTensorArgument(std::string_view name) const {
  return FindTensorArgument(name);
}


bool OpSchema::ArgSupportsPerFrameInput(std::string_view arg_name) const {
  return FindArgument(arg_name, [](const ArgumentDef &arg) { return arg.per_frame; });
}


const ArgumentDef *OpSchema::FindTensorArgument(std::string_view name) const {
  return FindArgument(name, [](const ArgumentDef &def) { return def.tensor; });
}

void OpSchema::CheckInputIndex(int index) const {
  if (index < 0 && index >= max_num_input_)
    throw std::out_of_range(make_string(
        "Input index ", index, " is out of range [0.." ,max_num_input_, ").\nWas NumInput called?");
}


std::string OpSchema::DefaultDeprecatedArgMsg(std::string_view arg_name,
                                              std::string_view renamed_to, bool removed) const {
  std::stringstream ss;
  if (removed) {
    ss << "The argument `" << arg_name
       << "` is no longer used and will be removed in a future release.";
  } else if (!renamed_to.empty()) {
    ss << "The argument `" << arg_name << "` is a deprecated alias for `" << renamed_to
       << "`. Use `" << renamed_to << "` instead.";
  } else {
    ss << "The argument `" << arg_name << "` is now deprecated and its usage is discouraged.";
  }
  return ss.str();
}


const Value *OpSchema::FindDefaultValue(std::string_view name) const {
  if (auto *arg = FindArgument(name))
    return arg->default_value.get();
  else
    return nullptr;
}


bool OpSchema::HasArgument(std::string_view name, bool include_internal) const {
  if (auto *arg = FindArgument(name))
    return arg && (include_internal || !arg.internal);
  else
    return false;
}

}  // namespace dali
