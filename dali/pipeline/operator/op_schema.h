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

#ifndef DALI_PIPELINE_OPERATOR_OP_SCHEMA_H_
#define DALI_PIPELINE_OPERATOR_OP_SCHEMA_H_

#include <functional>
#include <map>
#include <string>
#include <set>
#include <vector>
#include <utility>
#include <memory>

#include "dali/core/common.h"
#include "dali/core/traits.h"
#include "dali/core/error_handling.h"
#include "dali/pipeline/operator/argument.h"

namespace dali {

class OpSpec;

class DLL_PUBLIC OpSchema {
 public:
  typedef std::function<int(const OpSpec &spec)> SpecFunc;

  OpSchema(OpSchema &&) = default;
  OpSchema(const OpSchema &) = delete;
  OpSchema &operator=(const OpSchema &) = delete;
  OpSchema &operator=(OpSchema &&) = default;

  DLL_PUBLIC explicit inline OpSchema(const std::string &name): name_(name) {
    // Fill internal arguments
    auto v = Value::construct(-1);
    internal_arguments_["num_threads"] =
        std::make_pair("Number of CPU threads in a thread pool", v.get());
    internal_arguments_unq_.push_back(std::move(v));
    v = Value::construct(-1);
    internal_arguments_["batch_size"] = std::make_pair("Batch size", v.get());
    internal_arguments_unq_.push_back(std::move(v));
    v = Value::construct(1);
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

    AddOptionalArg("preserve", "Do not remove the Op from the "
                               "graph even if its outputs are unused.", false);
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
    output_fn_ = std::move(f);
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
    additional_outputs_fn_ = std::move(f);
    return *this;
  }

  /**
   * @brief Sets the number of inputs that the op can receive.
   */
  DLL_PUBLIC inline OpSchema& NumInput(int n) {
    DALI_ENFORCE(n >= 0);
    max_num_input_ = n;
    min_num_input_ = n;
    input_layouts_.resize(n);
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
    input_layouts_.resize(max);
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
   * @brief Indicates that multiple instances of this operator cannot share a logical ID to achieve
   * uniform processing of multiple input sets
   */
  DLL_PUBLIC inline OpSchema& DisallowInstanceGrouping() {
    allow_instance_grouping_ = false;
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
   * @brief Notes that this operator is deprecated and optionally specifies the operator to be used instead
   */
  DLL_PUBLIC inline OpSchema& Deprecate(const std::string &in_favor_of) {
    is_deprecated_ = true;
    deprecated_in_favor_of_ = in_favor_of;
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

  /**
   * @brief Sets input layout constraints and default for given input.
   *
   * At run-time, when the operator encounters a tensor(list) with specified
   * layout, but different than one provided to this function, error is raised.
   *
   * If the input tensor has no layout, the one provided to this function is assumed
   * if number of dimensions matches. Otherswise, error is raised.
   */
  DLL_PUBLIC inline OpSchema& InputLayout(int index, TensorLayout layout) {
    return InputLayout(index, { layout });
  }

  /**
   * @brief Sets input layout constraints and default for given input.
   *
   * At run-time, when the operator encounters a tensor(list) with specified
   * layout, but not one of those provided to this function, error is raised.
   *
   * If the input tensor has no layout, the layouts specified by call to this function
   * are searched for the first matching the number of dimensions of the input -
   * it will be the default value for this input. If number of dimensions doesn't
   * match any of the layouts provided here, an error is raised.
   */
  DLL_PUBLIC inline OpSchema& InputLayout(int index, std::initializer_list<TensorLayout> layouts) {
    CheckInputIndex(index);
    DALI_ENFORCE(input_layouts_[index].empty(), "Layouts for input " + std::to_string(index) +
                 " already specified");
    for (auto &l : layouts) {
      DALI_ENFORCE(!l.empty(), "Cannot specify an empty layout for an input");
    }
    input_layouts_[index] = layouts;
    return *this;
  }

  /**
   * @brief Sets input layout constraint and default for all inputs.
   * @see InputLayout(int index, TensorLayout layout)
   */
  DLL_PUBLIC inline OpSchema& InputLayout(TensorLayout layout) {
    return InputLayout({ layout });
  }

  /**
   * @brief Sets input layout constraint and default for all inputs.
   * @see InputLayout(int index, TensorLayout layout)
   */
  DLL_PUBLIC inline OpSchema& InputLayout(std::initializer_list<TensorLayout> layouts) {
    for (int i = 0; i < max_num_input_; i++)
      InputLayout(i, layouts);
    return *this;
  }

  /**
   * @brief Verifies that the layout is valid for given input index and number of dimensions
   *        or returns a default layout if the layout parameter is empty.
   */
  DLL_PUBLIC inline const TensorLayout &GetInputLayout(int index, int sample_ndim,
                                                       const TensorLayout &layout = {}) const {
    CheckInputIndex(index);
    if (input_layouts_[index].empty()) {
      DALI_ENFORCE(layout.empty() || layout.ndim() == sample_ndim,
        "The layout for the input has different number of dimensions than actual input");
      return layout;
    }

    if (layout.empty()) {
      for (auto &l : input_layouts_[index])
        if (l.ndim() == sample_ndim)
          return l;
      std::stringstream ss;
      ss << "The number of dimensions " << sample_ndim << " does not match any of the allowed"
        " layouts for input " << index << ". Valid layouts are:\n";
      for (auto &l : input_layouts_[index])
        ss << l.c_str() << "\n";
      DALI_FAIL(ss.str());
    } else {
      for (auto &l : input_layouts_[index])
        if (l == layout)
          return l;
      std::stringstream ss;
      ss << "The layout \"" << layout.c_str() << "\" does not match any of the allowed"
        " layouts for input " << index << ". Valid layouts are:\n";
      for (auto &l : input_layouts_[index])
        ss << l.c_str() << "\n";
      DALI_FAIL(ss.str());
    }
  }

  /**
   * @brief Adds an optional non-vector argument to op
   */
  template <typename T>
  DLL_PUBLIC inline std::enable_if_t<!is_vector<T>::value && !is_std_array<T>::value, OpSchema&>
  AddOptionalArg(const std::string &s,
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

  DLL_PUBLIC inline OpSchema &AddOptionalArg(const std::string &s,
                                             const std::string &doc,
                                             const char *default_value) {
    return AddOptionalArg(s, doc, std::string(default_value), false);
  }

  /**
   * @brief Adds an optional vector argument to op
   */
  template <typename T>
  DLL_PUBLIC inline OpSchema& AddOptionalArg(const std::string &s, const std::string &doc,
                                             std::vector<T> default_value,
                                             bool enable_tensor_input = false) {
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
    (void)f;
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

  /**
   * @brief Notes that this operator should not be pruned from
   * a graph even if its outputs are unused.
   */
  DLL_PUBLIC inline OpSchema& NoPrune() {
    no_prune_ = true;
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

  DLL_PUBLIC inline bool AllowsInstanceGrouping() const {
    return allow_instance_grouping_;
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

  DLL_PUBLIC inline bool IsDeprecated() const {
    return is_deprecated_;
  }

  DLL_PUBLIC inline const std::string& DeprecatedInFavorOf() const {
    return deprecated_in_favor_of_;
  }

  DLL_PUBLIC inline bool HasOutputFn() const {
    return static_cast<bool>(output_fn_);
  }

  DLL_PUBLIC inline bool IsNoPrune() const {
    return no_prune_;
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

  template<typename T>
  DLL_PUBLIC inline T GetDefaultValueForOptionalArgument(const std::string &s) const;

  DLL_PUBLIC bool HasRequiredArgument(const std::string &name, const bool local_only = false) const;

  DLL_PUBLIC bool HasOptionalArgument(const std::string &name, const bool local_only = false) const;

  /**
   * @brief Finds default value for a given argument
   * @return A pair of the defining schema and the value
   */
  DLL_PUBLIC std::pair<const OpSchema *, const Value *>
  FindDefaultValue(const std::string &arg_name,
                   bool local_only = false,
                   bool include_internal = true) const;

  DLL_PUBLIC inline bool HasArgument(const std::string &name) const {
    return HasRequiredArgument(name) || HasOptionalArgument(name);
  }

  DLL_PUBLIC std::string GetArgumentDox(const std::string &name) const;
  DLL_PUBLIC DALIDataType GetArgumentType(const std::string &name) const;
  DLL_PUBLIC std::string GetArgumentDefaultValueString(const std::string &name) const;
  DLL_PUBLIC std::vector<std::string> GetArgumentNames() const;
  DLL_PUBLIC bool IsTensorArgument(const std::string &name) const;

 private:
  inline void CheckArgument(const std::string &s) {
    DALI_ENFORCE(!HasArgument(s),
                 "Argument \"" + s + "\" already added to the schema");
    DALI_ENFORCE(internal_arguments_.find(s) == internal_arguments_.end(),
                 "Argument name \"" + s + "\" is reserved for internal use");
  }

  inline void CheckInputIndex(int index) const {
    DALI_ENFORCE(index >= 0 && index < max_num_input_,
      "Output index (=" + std::to_string(index) +  ") out of range [0.." +
      std::to_string(max_num_input_) + ").\nWas NumInput called?");
  }

  std::map<std::string, std::pair<std::string, DALIDataType> > GetRequiredArguments() const;

  std::map<std::string, std::pair<std::string, Value*>> GetOptionalArguments() const;

  string dox_;
  string name_;
  SpecFunc output_fn_, in_place_fn_, additional_outputs_fn_;

  int min_num_input_ = 0, max_num_input_ = 0;
  int num_output_ = 0;

  bool allow_instance_grouping_ = true;
  vector<string> parents_;

  bool allow_sequences_ = false;
  bool is_sequence_operator_ = false;

  bool is_internal_ = false;

  bool no_prune_ = false;

  bool is_deprecated_ = false;
  string deprecated_in_favor_of_;

  std::map<std::string, std::pair<std::string, DALIDataType> > arguments_;
  std::map<std::string, std::pair<std::string, Value*> > optional_arguments_;
  std::map<std::string, std::pair<std::string, Value*> > internal_arguments_;
  std::vector<std::unique_ptr<Value> > optional_arguments_unq_;
  std::vector<std::unique_ptr<Value> > internal_arguments_unq_;
  std::vector<std::vector<TensorLayout>> input_layouts_;

  std::set<std::string> tensor_arguments_;
};

class SchemaRegistry {
 public:
  DLL_PUBLIC static OpSchema& RegisterSchema(const std::string &name);
  DLL_PUBLIC static const OpSchema& GetSchema(const std::string &name);
  DLL_PUBLIC static const OpSchema* TryGetSchema(const std::string &name);

 private:
  inline SchemaRegistry() {}

  DLL_PUBLIC static std::map<string, OpSchema>& registry();
};

template<typename T>
inline T OpSchema::GetDefaultValueForOptionalArgument(const std::string &s) const {
  const Value *v = FindDefaultValue(s, false, true).second;
  DALI_ENFORCE(v != nullptr, "Optional argument \"" + s + "\" is not defined for schema \""
      + this->name() + "\"");

  const ValueInst<T> * vT = dynamic_cast<const ValueInst<T>*>(v);
  DALI_ENFORCE(vT != nullptr, "Unexpected type of the default value for argument \"" + s +
        "\" of schema \"" + this->name() + "\"");
  return vT->Get();
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

#endif  // DALI_PIPELINE_OPERATOR_OP_SCHEMA_H_
