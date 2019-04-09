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

#ifndef DALI_PIPELINE_OPERATORS_OP_SCHEMA_H_
#define DALI_PIPELINE_OPERATORS_OP_SCHEMA_H_

#include <functional>
#include <map>
#include <string>
#include <set>
#include <vector>
#include <utility>
#include <memory>

#include "dali/common.h"
#include "dali/error_handling.h"
#include "dali/pipeline/operators/argument.h"

namespace dali {

class OpSpec;

class DLL_PUBLIC OpSchema {
 public:
  typedef std::function<int(const OpSpec &spec)> SpecFunc;

  OpSchema(OpSchema &&) = default;
  OpSchema(const OpSchema &) = delete;
  OpSchema &operator=(const OpSchema &) = delete;
  OpSchema &operator=(OpSchema &&) = default;

  DLL_PUBLIC explicit inline OpSchema(const std::string &name)
    : name_(name)
    , allow_multiple_input_sets_(false)
    , enforce_layout_(false)
    , allow_sequences_(false)
    , is_sequence_operator_(false)
    , is_internal_(false) {
    // Fill internal arguments
    auto v = Value::construct(-1);
    internal_arguments_["num_threads"] =
        std::make_pair("Number of CPU threads in a thread pool", v.get());
    internal_arguments_unq_.push_back(std::move(v));
    v = Value::construct(-1);
    internal_arguments_["batch_size"] = std::make_pair("Batch size", v.get());
    internal_arguments_unq_.push_back(std::move(v));
    v = Value::construct(1);
    internal_arguments_["num_input_sets"] =
        std::make_pair("Number of input sets given to an Op", v.get());
    internal_arguments_unq_.push_back(std::move(v));
    v = Value::construct(std::string("cpu"));
    internal_arguments_["device"] = std::make_pair("Device on which the Op is run", v.get());
    internal_arguments_unq_.push_back(std::move(v));
    v = Value::construct(false);
    internal_arguments_["inplace"] = std::make_pair("Whether Op can be run in place", v.get());
    internal_arguments_unq_.push_back(std::move(v));
    v = Value::construct(0);
    internal_arguments_["default_cuda_stream_priority"] =
        std::make_pair("Default cuda stream priority", v.get());
    internal_arguments_unq_.push_back(std::move(v));

    AddOptionalArg("seed", "Random seed (If not provided it will be populated based "
      "on the global seed of the pipeline)", -1);

    AddOptionalArg("bytes_per_sample_hint", "Output size hint (bytes), "
      "per sample. The memory will be preallocated if it uses GPU or page-locked memory", 0);
  }

  DLL_PUBLIC inline ~OpSchema() = default;

  /**
   * @brief Returns the name of this operator.
   */
  DLL_PUBLIC inline const std::string& name() const {
    return name_;
  }

  /**
   * @brief Sets the doc string for this operator.
   */
  DLL_PUBLIC inline OpSchema& DocStr(const string &dox) {
    dox_ = dox;
    return *this;
  }

  /**
   * @brief Sets a funtion that infers the number of outputs this
   * op will produce from the ops specfication. This is required
   * to expose the op to the python interface.
   *
   * If the ops has a fixed number of outputs, this function
   * does not need to be added to the schema
   */
  DLL_PUBLIC inline OpSchema& OutputFn(SpecFunc f) {
    output_fn_ = f;
    return *this;
  }

  /**
   * @brief Sets a function to determine the number of
   * additional outputs (independent of output sets) from an
   * op from the op's specification.
   *
   * If this function is not set it will be assumed that no
   * additional outputs can be returned
   *
   * Use case is to expose additional information (such as random
   * numbers used within operators) to the user
   */
  DLL_PUBLIC inline OpSchema& AdditionalOutputsFn(SpecFunc f) {
    additional_outputs_fn_ = f;
    return *this;
  }

  /**
   * @brief Sets the number of inputs that the op can receive.
   */
  DLL_PUBLIC inline OpSchema& NumInput(int n) {
    DALI_ENFORCE(n >= 0);
    max_num_input_ = n;
    min_num_input_ = n;
    return *this;
  }

  /**
   * @brief Sets the min and max number of inputs the op can receive.
   */
  DLL_PUBLIC inline OpSchema& NumInput(int min, int max) {
    DALI_ENFORCE(min <= max);
    DALI_ENFORCE(min >= 0);
    DALI_ENFORCE(max >= 0);
    min_num_input_ = min;
    max_num_input_ = max;
    return *this;
  }

  /**
   * @brief Sets the number of outputs that the op can receive.
   */
  DLL_PUBLIC inline OpSchema& NumOutput(int n) {
    DALI_ENFORCE(n >= 0);
    num_output_ = n;
    return *this;
  }

  /**
   * @brief Notes that multiple input sets can be used with this op
   */
  DLL_PUBLIC inline OpSchema& AllowMultipleInputSets() {
    allow_multiple_input_sets_ = true;
    return *this;
  }

  /**
   * @brief Notes that this operator expects sequence inputs exclusively
   */
  DLL_PUBLIC inline OpSchema& SequenceOperator() {
    is_sequence_operator_ = true;
    return *this;
  }

  /**
   * @brief Notes that sequences can be used with this op
   */
  DLL_PUBLIC inline OpSchema& AllowSequences() {
    allow_sequences_ = true;
    return *this;
  }

  /**
   * @brief Notes that this operator is internal to DALI backend (and shouldn't be exposed in Python API)
   */
  DLL_PUBLIC inline OpSchema& MakeInternal() {
    is_internal_ = true;
    return *this;
  }

  /**
   * @brief Adds a required argument to op with its type
   */
  DLL_PUBLIC inline OpSchema& AddArg(const std::string &s,
                                     const std::string &doc,
                                     const DALIDataType dtype,
                                     bool enable_tensor_input = false) {
    CheckArgument(s);
    arguments_[s] = std::make_pair(doc, dtype);
    if (enable_tensor_input) {
      tensor_arguments_.insert(s);
    }
    return *this;
  }

  DLL_PUBLIC inline OpSchema& EnforceInputLayout(DALITensorLayout layout) {
    layout_ = layout;
    enforce_layout_ = true;
    return *this;
  }

  /**
   * @brief Adds an optional non-vector argument to op
   */
  template <typename T>
  DLL_PUBLIC inline typename std::enable_if<
    !is_vector<T>::value && !is_std_array<T>::value,
    OpSchema&>::type AddOptionalArg(const std::string &s,
                                    const std::string &doc,
                                    T default_value,
                                    bool enable_tensor_input = false) {
    CheckArgument(s);
    auto to_store = Value::construct(default_value);
    optional_arguments_[s] = std::make_pair(doc, to_store.get());
    optional_arguments_unq_.push_back(std::move(to_store));
    if (enable_tensor_input) {
      tensor_arguments_.insert(s);
    }
    return *this;
  }

  /**
   * @brief Adds an optional vector argument to op
   */
  template <typename T>
  DLL_PUBLIC inline OpSchema& AddOptionalArg(const std::string &s, const std::string &doc,
                                  std::vector<T> default_value, bool enable_tensor_input = false) {
    CheckArgument(s);
    auto to_store = Value::construct(std::vector<T>(default_value));
    optional_arguments_[s] = std::make_pair(doc, to_store.get());
    optional_arguments_unq_.push_back(std::move(to_store));
    if (enable_tensor_input) {
      tensor_arguments_.insert(s);
    }
    return *this;
  }

  /**
   * @brief Sets a function that infers whether the op can
   * be executed in-place depending on the ops specification.
   */
  DLL_PUBLIC inline OpSchema& InPlaceFn(SpecFunc f) {
    REPORT_FATAL_PROBLEM("In-place op support not yet implemented.");
    return *this;
  }

  /**
   * @brief Sets a parent (which could be used as a storage of default parameters)
   * Does not support cyclic dependency.
   */
  DLL_PUBLIC inline OpSchema& AddParent(const std::string &parentName) {
    parents_.push_back(parentName);
    return *this;
  }

  DLL_PUBLIC inline OpSchema& SetName(const std::string &name) {
    name_ = name;
    return *this;
  }

  DLL_PUBLIC inline const vector<std::string>& GetParents() const {
    return parents_;
  }

  DLL_PUBLIC string Dox() const;

  DLL_PUBLIC inline int MaxNumInput() const {
    return max_num_input_;
  }

  DLL_PUBLIC inline int MinNumInput() const {
    return min_num_input_;
  }

  DLL_PUBLIC inline int NumOutput() const {
    return num_output_;
  }

  DLL_PUBLIC inline bool AllowsMultipleInputSets() const {
    return allow_multiple_input_sets_;
  }

  DLL_PUBLIC inline bool EnforceInputLayout() const {
    return enforce_layout_;
  }

  DLL_PUBLIC inline DALITensorLayout InputLayout() const {
    return layout_;
  }

  DLL_PUBLIC inline bool IsSequenceOperator() const {
    return is_sequence_operator_;
  }

  DLL_PUBLIC inline bool AllowsSequences() const {
    return allow_sequences_;
  }

  DLL_PUBLIC inline bool IsInternal() const {
    return is_internal_;
  }

  DLL_PUBLIC inline bool HasOutputFn() const {
    return static_cast<bool>(output_fn_);
  }

  DLL_PUBLIC int CalculateOutputs(const OpSpec &spec) const;

  DLL_PUBLIC int CalculateAdditionalOutputs(const OpSpec &spec) const {
    if (!additional_outputs_fn_) return 0;
    return additional_outputs_fn_(spec);
  }

  DLL_PUBLIC inline bool SupportsInPlace(const OpSpec &spec) const {
    if (!in_place_fn_) return false;
    return in_place_fn_(spec);
  }

  DLL_PUBLIC void CheckArgs(const OpSpec &spec) const;

  DLL_PUBLIC inline const OpSchema& GetSchemaWithArgument(const string& name) const;

  template<typename T>
  DLL_PUBLIC inline T GetDefaultValueForOptionalArgument(const std::string &s) const;

  DLL_PUBLIC bool HasRequiredArgument(const std::string &name, const bool local_only = false) const;

  DLL_PUBLIC bool HasOptionalArgument(const std::string &name, const bool local_only = false) const;

  DLL_PUBLIC inline bool HasArgument(const std::string &name) const {
    return HasRequiredArgument(name) || HasOptionalArgument(name);
  }

  DLL_PUBLIC std::string GetArgumentDox(const std::string &name) const;
  DLL_PUBLIC DALIDataType GetArgumentType(const std::string &name) const;
  DLL_PUBLIC std::string GetArgumentDefaultValueString(const std::string &name) const;
  DLL_PUBLIC std::vector<std::string> GetArgumentNames() const;
  DLL_PUBLIC bool IsTensorArgument(const std::string &name) const;

 private:
  inline bool CheckArgument(const std::string &s) {
    DALI_ENFORCE(!HasArgument(s),
                 "Argument \"" + s + "\" already added to the schema");
    DALI_ENFORCE(internal_arguments_.find(s) == internal_arguments_.end(),
                 "Argument name \"" + s + "\" is reserved for internal use");
    return true;
  }

  std::map<std::string, std::pair<std::string, DALIDataType> > GetRequiredArguments() const;

  std::map<std::string, std::pair<std::string, Value*>> GetOptionalArguments() const;

  string dox_;
  string name_;
  SpecFunc output_fn_, in_place_fn_, additional_outputs_fn_;

  int min_num_input_ = 0, max_num_input_ = 0;
  int num_output_ = 0;

  bool allow_multiple_input_sets_;
  vector<string> parents_;

  bool enforce_layout_;
  DALITensorLayout layout_;

  bool allow_sequences_;
  bool is_sequence_operator_;

  bool is_internal_;

  std::map<std::string, std::pair<std::string, DALIDataType> > arguments_;
  std::map<std::string, std::pair<std::string, Value*> > optional_arguments_;
  std::map<std::string, std::pair<std::string, Value*> > internal_arguments_;
  std::vector<std::unique_ptr<Value> > optional_arguments_unq_;
  std::vector<std::unique_ptr<Value> > internal_arguments_unq_;

  std::set<std::string> tensor_arguments_;
};

class SchemaRegistry {
 public:
  static OpSchema& RegisterSchema(const std::string &name) {
    auto &schema_map = registry();
    DALI_ENFORCE(schema_map.count(name) == 0, "OpSchema already "
        "registered for operator '" + name + "'. DALI_SCHEMA(op) "
        "should only be called once per op.");

    // Insert the op schema and return a reference to it
    schema_map.emplace(std::make_pair(name, OpSchema(name)));
    return schema_map.at(name);
  }

  static const OpSchema& GetSchema(const std::string &name) {
    auto &schema_map = registry();
    auto it = schema_map.find(name);
    DALI_ENFORCE(it != schema_map.end(), "Schema for operator '" +
        name + "' not registered");
    return it->second;
  }

 private:
  inline SchemaRegistry() {}

  DLL_PUBLIC static std::map<string, OpSchema>& registry();
};

inline string GetSchemaWithArg(const string& start, const string& arg) {
  const OpSchema& s = SchemaRegistry::GetSchema(start);

  // Found locally, return immediately
  if (s.HasOptionalArgument(arg, true)) {
    return start;
  }
  // otherwise, loop over any parents
  for (auto& parent : s.GetParents()) {
    // recurse
    string tmp = GetSchemaWithArg(parent, arg);

    // we found the schema, return
    if (!tmp.empty()) {
      return tmp;
    }
  }
  // default case, return empty string
  return string{};
}

template<typename T>
inline T OpSchema::GetDefaultValueForOptionalArgument(const std::string &s) const {
  // check if argument exists in this schema
  const bool argFound = HasOptionalArgument(s, true);

  if (argFound || internal_arguments_.find(s) != internal_arguments_.end()) {
    Value* v;
    if (argFound) {
      auto arg_pair = *optional_arguments_.find(s);
      v = arg_pair.second.second;
    } else {
      auto arg_pair = *internal_arguments_.find(s);
      v = arg_pair.second.second;
    }
    ValueInst<T> * vT = dynamic_cast<ValueInst<T>*>(v);
    DALI_ENFORCE(vT != nullptr, "Unexpected type of the default value for argument \"" + s +
         "\" of schema \"" + this->name() + "\"");
    return vT->Get();
  } else {
    // get the parent schema that has the optional argument and return from there
    string tmp = GetSchemaWithArg(name(), s);
    DALI_ENFORCE(!tmp.empty(), "Optional argument \"" + s + "\" is not defined for schema \""
        + this->name() + "\"");

    const OpSchema& schema = SchemaRegistry::GetSchema(tmp);
    return schema.template GetDefaultValueForOptionalArgument<T>(s);
  }
}

#define DALI_SCHEMA_REG(OpName)      \
  int DALI_OPERATOR_SCHEMA_REQUIRED_FOR_##OpName() {        \
    return 42;                                              \
  }                                                         \
  static ::dali::OpSchema* ANONYMIZE_VARIABLE(OpName) =             \
    &::dali::SchemaRegistry::RegisterSchema(#OpName)

#define DALI_SCHEMA(OpName)                            \
      DALI_SCHEMA_REG(OpName)

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_OP_SCHEMA_H_
