// Copyright (c) 2017-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/core/copy_vector_helper.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/traits.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/argument.h"

namespace dali {

class OpSpec;

struct RequiredArgumentDef {
  std::string doc;
  DALIDataType dtype;
};

struct DefaultedArgumentDef {
  std::string doc;
  DALIDataType dtype;
  Value *default_value;
  // As opposed to purely internal argument, the hidden argument
  // can be specified as any other argument on per-operator basis, through Python API, etc.
  // It is just hidden from the docs.
  bool hidden;
};

struct DeprecatedArgDef {
  std::string renamed_to = {};
  std::string msg = {};
  bool removed = false;
};

struct TensorArgDesc {
  bool supports_per_frame = false;
};

enum class InputDevice : uint8_t {
  MatchBackend = 0,
  CPU = 1,
  GPU = 2,
};

class DLL_PUBLIC OpSchema {
 public:
  typedef std::function<int(const OpSpec &spec)> SpecFunc;

  OpSchema(OpSchema &&) = default;
  OpSchema(const OpSchema &) = delete;
  OpSchema &operator=(const OpSchema &) = delete;
  OpSchema &operator=(OpSchema &&) = default;

  DLL_PUBLIC explicit OpSchema(const std::string &name);

  DLL_PUBLIC inline ~OpSchema() = default;

  /**
   * @brief Returns the name of this operator.
   */
  DLL_PUBLIC const std::string &name() const;

  /**
   * @brief Sets the doc string for this operator.
   */
  DLL_PUBLIC OpSchema &DocStr(const string &dox);

  /**
   * @brief Sets the docstring for input.
   *
   * Set the documentation for intput at given `index`.
   *
   * If the operator specifies some range of allowed inputs with NumInput(int min, int max)
   * only the first `min` inputs are considered mandatory, the rest are optional
   *
   * Will generate entry in `Args` section using numpydoc style:
   * `name`: type_doc
   *     doc
   */
  DLL_PUBLIC OpSchema &InputDox(int index, const string &name, const string &type_doc,
                                const string &doc);

  /**
   * @brief Allows to set a docstring for __call__ method of Operator.
   *
   * The first line of the string can contain the signature that will be used
   * in the sphinx-generated documentation, for example:
   * "__call__(input0, input1, optional_input = None, **kwargs)\n"
   *
   * The arguments should be described using Args section and numpydoc syntax,
   * with comments indented by 4 spaces, for example:
   * """
   * Args
   * ----
   * `input0`: Type of input
   *     This is the first input
   * `input1`: TensorList of some kind
   *     This is second input
   * `optional_input`: TensorList, optional
   *     This is optional input
   *
   * If the `append_kwargs_section` is true, the docstring generator will append the Keyword args
   * section at the end of this doc
   *
   * @param doc
   * @param append_kwargs_section
   */
  DLL_PUBLIC OpSchema &CallDocStr(const string &doc, bool append_kwargs_section = false);

  /**
   * @brief Sets a function that infers the number of outputs this
   * op will produce from the ops specification. This is required
   * to expose the op to the python interface.
   *
   * If the ops has a fixed number of outputs, this function
   * does not need to be added to the schema
   */
  DLL_PUBLIC OpSchema &OutputFn(SpecFunc f);

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
  DLL_PUBLIC OpSchema &AdditionalOutputsFn(SpecFunc f);

  /**
   * @brief Sets the number of inputs that the op can receive.
   */
  DLL_PUBLIC OpSchema &NumInput(int n);

  /**
   * @brief Sets the min and max number of inputs the op can receive.
   */
  DLL_PUBLIC OpSchema &NumInput(int min, int max);

  /**
   * @brief Sets the input device for given range of inputs
   */
  DLL_PUBLIC OpSchema &InputDevice(int first, int one_past, dali::InputDevice device);

  /**
   * @brief Sets the input device for given range of input
   */
  DLL_PUBLIC OpSchema &InputDevice(int index, dali::InputDevice device);

  /**
   * @brief Gets the supported input device for given input
   */
  DLL_PUBLIC dali::InputDevice GetInputDevice(int index) const;

  /**
   * @brief Sets the number of outputs that the op can receive.
   */
  DLL_PUBLIC OpSchema &NumOutput(int n);

  /**
   * @brief Indicates that this operator should not use auto-generated documentation
   *        of inputs and `__call__` operator with custom signature.
   */
  DLL_PUBLIC OpSchema &DisableAutoInputDox();

  /**
   * @brief Indicates that multiple instances of this operator cannot share a logical ID to achieve
   * uniform processing of multiple input sets
   */
  DLL_PUBLIC OpSchema &DisallowInstanceGrouping();

  /**
   * @brief Notes that this operator expects sequence inputs exclusively
   */
  DLL_PUBLIC OpSchema &SequenceOperator();

  /**
   * @brief Notes that sequences can be used with this op
   */
  DLL_PUBLIC OpSchema &AllowSequences();

  /**
   * Notes that the operator can process 3D data.
   * @return
   */
  DLL_PUBLIC OpSchema &SupportVolumetric();

  /**
   * @brief Notes that this operator is internal to DALI backend (and shouldn't be exposed in Python
   * API)
   */
  DLL_PUBLIC OpSchema &MakeInternal();

  /**
   * @brief Notes that this operator doc should not be visible (but the Op is exposed in Python API)
   */
  DLL_PUBLIC OpSchema &MakeDocHidden();

  /**
   * @brief Notes that for this operator only the doc_str should be visible, but not the docs for
   * the inputs, outputs or argument (the Op is exposed in Python API)
   */
  DLL_PUBLIC OpSchema &MakeDocPartiallyHidden();

  /**
   * @brief Notes that this operator is deprecated and optionally specifies the operator to be used
   * instead
   *
   * @param in_favor_of schema name of the replacement
   * @param explanation additional explanation
   */
  DLL_PUBLIC OpSchema &Deprecate(const std::string &in_favor_of,
                                 const std::string &explanation = "");

  /**
   * @brief Notes that this operator cannot be serialized
   */
  DLL_PUBLIC OpSchema &Unserializable();

  /**
   * @brief Adds a required argument to op with its type
   */
  DLL_PUBLIC OpSchema &AddArg(const std::string &s, const std::string &doc,
                              const DALIDataType dtype, bool enable_tensor_input = false,
                              bool support_per_frame_input = false);


  /**
   * @brief Adds a required argument of type DALIDataType
   */
  DLL_PUBLIC OpSchema &AddTypeArg(const std::string &s, const std::string &doc);

  /**
   * @brief Sets input layout constraints and default for given input.
   *
   * At run-time, when the operator encounters a tensor(list) with specified
   * layout, but different than one provided to this function, error is raised.
   *
   * If the input tensor has no layout, the one provided to this function is assumed
   * if number of dimensions matches. Otherswise, error is raised.
   */
  DLL_PUBLIC OpSchema &InputLayout(int index, TensorLayout layout);

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
  DLL_PUBLIC OpSchema &InputLayout(int index, std::initializer_list<TensorLayout> layouts);

  /**
   * @brief Sets input layout constraint and default for all inputs.
   * @see InputLayout(int index, TensorLayout layout)
   */
  DLL_PUBLIC OpSchema &InputLayout(TensorLayout layout);

  /**
   * @brief Sets input layout constraint and default for all inputs.
   * @see InputLayout(int index, TensorLayout layout)
   */
  DLL_PUBLIC OpSchema &InputLayout(std::initializer_list<TensorLayout> layouts);

  /**
   * @brief Verifies that the layout is valid for given input index and number of dimensions
   *        or returns a default layout if the layout parameter is empty.
   */
  DLL_PUBLIC const TensorLayout &GetInputLayout(int index, int sample_ndim,
                                                const TensorLayout &layout = {}) const;


  DLL_PUBLIC const std::vector<TensorLayout> &GetSupportedLayouts(int input_idx) const;

  /**
   * @brief Adds an optional non-vector argument without default to op
   *        The type can be specified as enum, nullptr_t is used for overload resolution
   *        If the arg name starts is with an underscore, it will be marked hidden, which
   *        makes it not listed in the docs.
   */
  DLL_PUBLIC OpSchema &AddOptionalArg(const std::string &s, const std::string &doc,
                                      DALIDataType dtype, std::nullptr_t,
                                      bool enable_tensor_input = false,
                                      bool support_per_frame_input = false);

  /**
   * @brief Adds an optional non-vector argument without default to op.
   *        If the arg name starts is with an underscore, it will be marked hidden, which
   *        makes it not listed in the docs.
   */
  template <typename T>
  DLL_PUBLIC inline OpSchema &AddOptionalArg(const std::string &s, const std::string &doc,
                                             std::nullptr_t, bool enable_tensor_input = false,
                                             bool support_per_frame_input = false) {
    AddOptionalArg(s, doc, type2id<T>::value, nullptr, enable_tensor_input,
                   support_per_frame_input);
    return *this;
  }


  /**
   * @brief Adds an optional non-vector argument to op
   *
   *        If the arg name starts is with an underscore, it will be marked hidden, which
   *        makes it not listed in the docs.
   */
  template <typename T>
  DLL_PUBLIC inline std::enable_if_t<!is_vector<T>::value && !is_std_array<T>::value, OpSchema &>
  AddOptionalArg(const std::string &s, const std::string &doc, T default_value,
                 bool enable_tensor_input = false, bool support_per_frame_input = false) {
    static_assert(
        !std::is_same<T, DALIDataType>::value,
        R"(Use `AddOptionalTypeArg` instead. `AddOptionalArg` with a default value should not be
used with DALIDataType, to avoid confusion with `AddOptionalArg<type>(name, doc, nullptr)`)");
    CheckArgument(s);
    auto to_store = Value::construct(default_value);
    optional_arguments_[s] = {doc, type2id<T>::value, to_store.get(), ShouldHideArgument(s)};
    optional_arguments_unq_.push_back(std::move(to_store));
    if (enable_tensor_input) {
      tensor_arguments_[s] = {support_per_frame_input};
    }
    return *this;
  }

  /**
   * @brief Adds an optional argument of type DALIDataType with a default value
   *
   *        If the arg name starts is with an underscore, it will be marked hidden, which
   *        makes it not listed in the docs.
   */
  DLL_PUBLIC OpSchema &AddOptionalTypeArg(const std::string &s, const std::string &doc,
                                          DALIDataType default_value);

  /**
   * @brief Adds an optional argument of type DALIDataType without a default value
   *
   *        If the arg name starts is with an underscore, it will be marked hidden, which
   *        makes it not listed in the docs.
   */
  DLL_PUBLIC OpSchema &AddOptionalTypeArg(const std::string &s, const std::string &doc);

  DLL_PUBLIC OpSchema &AddOptionalArg(const std::string &s, const std::string &doc,
                                      const char *default_value);

  /**
   * @brief Adds an optional vector argument to op
   *
   *        If the arg name starts is with an underscore, it will be marked hidden, which
   *        makes it not listed in the docs.
   */
  template <typename T>
  DLL_PUBLIC inline OpSchema &AddOptionalArg(const std::string &s, const std::string &doc,
                                             std::vector<T> default_value,
                                             bool enable_tensor_input = false,
                                             bool support_per_frame_input = false) {
    CheckArgument(s);
    using S = argument_storage_t<T>;
    auto to_store = Value::construct(detail::convert_vector<S>(default_value));
    bool hide_argument = ShouldHideArgument(s);
    optional_arguments_[s] = {doc, type2id<std::vector<T>>::value, to_store.get(), hide_argument};
    optional_arguments_unq_.push_back(std::move(to_store));
    if (enable_tensor_input) {
      tensor_arguments_[s] = {support_per_frame_input};
    }
    return *this;
  }

  /**
   * @brief Marks an argument as deprecated in favor of a new argument
   *
   * Providing renamed_to means the argument has been renamed and we can safely
   * propagate the value to the new argument name.
   */
  DLL_PUBLIC OpSchema &DeprecateArgInFavorOf(const std::string &arg_name, std::string renamed_to,
                                             std::string msg = {});

  /**
   * @brief Marks an argument as deprecated
   * @remarks There are three ways to deprecate an argument
   *          1. removed==true, means the operator will not use the
   *              argument at all and it can be safely discarded.
   *          2. removed==false, means the operator will still use the
   *              deprecated argument until it is finally removed completely from the schema.
   *          3. For renaming the argument see DeprecateArgInFavorOf
   */
  DLL_PUBLIC OpSchema &DeprecateArg(const std::string &arg_name, bool removed = true,
                                    std::string msg = {});

  /**
   * @brief Sets a function that infers whether the op can
   * be executed in-place depending on the ops specification.
   */
  DLL_PUBLIC OpSchema &InPlaceFn(SpecFunc f);

  /**
   * @brief Sets a parent (which could be used as a storage of default parameters)
   * Does not support cyclic dependency. There can be multiple parents
   * and the lookup is transitive.
   * Only arguments are inherited, inputs and outputs are not.
   */
  DLL_PUBLIC OpSchema &AddParent(const std::string &parentName);

  /**
   * @brief Notes that this operator should not be pruned from
   * a graph even if its outputs are unused.
   */
  DLL_PUBLIC OpSchema &NoPrune();

  /**
   * @brief Informs that the data passes through this operator unchanged, only
   *        the metadata is affected.
   *
   * If the operator _can_ pass an input buffer as-is to the output (possibly
   * changing the associated metadata), this property of should be set accordingly
   * in the operator's OpSchema. The purpose of this property is to inform the
   * pipeline and the executor that a particular output of the operator doesn't
   * own the storage and the associated input should be included in double-buffering
   * whenever the output should.
   *
   * @param inout - tells which inputs are passed through to which outputs.
   *                Only (partial - as in partial function) bijective mappings are allowed.
   */
  DLL_PUBLIC OpSchema &PassThrough(const std::map<int, int> &inout);

  /**
   * @brief Informs that the operator passes through data unchanged, sharing the allocation
   *        from input to output.
   *        The data is passed on sample basis, allowing to mix any input to any output.
   */
  DLL_PUBLIC OpSchema &SamplewisePassThrough();

  /**
   * @brief Get parent schemas (non-recursive)
   */
  DLL_PUBLIC const vector<std::string> &GetParents() const;

  /**
   * @brief Get the docstring of the operator - provided by DocStr in the schema definition.
   */
  DLL_PUBLIC string Dox() const;

  /**
   * @brief Return true wether the default input docs can be used
   */
  DLL_PUBLIC bool CanUseAutoInputDox() const;

  /**
   * @brief Whether the docstring for kwargs should be automatically generated and appended to the
   * one provided in CallDocStr.
   */
  DLL_PUBLIC bool AppendKwargsSection() const;

  /**
   * @brief Return true when `__call__` docstring was explicitly set
   *
   * Should be considered as highest preference
   */
  DLL_PUBLIC bool HasCallDox() const;

  /**
   * @brief Get the documentation for Operator __call__ signature provided by CallDocStr.
   */
  DLL_PUBLIC std::string GetCallDox() const;

  /**
   * @brief Check if this operator has input docstrings provided
   */
  DLL_PUBLIC bool HasInputDox() const;

  /**
   * @brief List all the inputs that should appear in `__call__` signature based on the input
   *        docs that were specified. Requires HasInputDox() to return true
   *
   */
  DLL_PUBLIC std::string GetCallSignatureInputs() const;

  /**
   * @brief Get the docstring name of the input at given index.
   */
  DLL_PUBLIC std::string GetInputName(int input_idx) const;

  /**
   * @brief Get the docstring type of the input at given index.
   */
  DLL_PUBLIC std::string GetInputType(int input_idx) const;

  /**
   * @brief Get the docstring text of the input at given index.
   */
  DLL_PUBLIC std::string GetInputDox(int input_idx) const;

  /**
   * @brief Get the maximal number of accepted inputs.
   */
  DLL_PUBLIC int MaxNumInput() const;

  /**
   * @brief Get the minimal number of required inputs.
   */
  DLL_PUBLIC int MinNumInput() const;

  /**
   * @brief Get the number of static outputs, see also CalculateOutputs and
   * CalculateAdditionalOutputs
   */
  DLL_PUBLIC int NumOutput() const;

  DLL_PUBLIC bool AllowsInstanceGrouping() const;

  /**
   * @brief Whether this operator accepts ONLY sequences as inputs
   */
  DLL_PUBLIC bool IsSequenceOperator() const;

  /**
   * @brief Whether this operator accepts sequences as inputs
   */
  DLL_PUBLIC bool AllowsSequences() const;

  /**
   * @brief Whether this operator accepts volumes as inputs
   */
  DLL_PUBLIC bool SupportsVolumetric() const;

  /**
   * @brief Whether this operator is internal to DALI backend (and shouldn't be exposed in Python
   * API)
   */
  DLL_PUBLIC bool IsInternal() const;

  /**
   * @brief Whether this operator doc should not be visible (but the Op is exposed in Python API)
   */
  DLL_PUBLIC bool IsDocHidden() const;

  /**
   * @brief Whether this operator doc should be visible without documenting any parameters
   * Useful for deprecated ops.
   */
  DLL_PUBLIC bool IsDocPartiallyHidden() const;

  /**
   * @brief Whether this operator is deprecated.
   */
  DLL_PUBLIC bool IsDeprecated() const;

  /**
   * @brief What operator replaced the current one.
   */
  DLL_PUBLIC const std::string &DeprecatedInFavorOf() const;

  /**
   * @brief Additional deprecation message
   */
  DLL_PUBLIC const std::string &DeprecationMessage() const;

  /**
   * @brief Whether given argument is deprecated.
   */
  DLL_PUBLIC bool IsDeprecatedArg(const std::string &arg_name) const;

  /**
   * @brief Metadata about the argument deprecation - error message, renaming, removal, etc.
   */
  DLL_PUBLIC const DeprecatedArgDef &DeprecatedArgMeta(const std::string &arg_name) const;

  /**
   * @brief Check whether this operator calculates number of outputs statically
   * @return false if static, true if dynamic
   */
  DLL_PUBLIC bool HasOutputFn() const;

  /**
   * @brief Check whether this operator won't be pruned out of graph even if not used.
   */
  DLL_PUBLIC bool IsNoPrune() const;

  DLL_PUBLIC bool IsSerializable() const;

  /**
   * @brief Returns the index of the output to which the input is passed.
   * @param strict consider only fully passed through batches
   * @return Output indicies or empty vector if given input is not passed through.
   */
  DLL_PUBLIC std::vector<int> GetPassThroughOutputIdx(int input_idx, const OpSpec &spec,
                                                      bool strict = true) const;

  /**
   * @brief Is the input_idx passed through to output_idx
   */
  DLL_PUBLIC bool IsPassThrough(int input_idx, int output_idx, bool strict = true) const;

  /**
   * @brief Does this operator pass through any data?
   */
  DLL_PUBLIC bool HasPassThrough() const;

  /**
   * @brief Does this operator pass through any data as a whole batch to batch?
   */
  DLL_PUBLIC bool HasStrictPassThrough() const;

  /**
   * @brief Does this operator pass through any data by the means of sharing individual samples?
   */
  DLL_PUBLIC bool HasSamplewisePassThrough() const;

  /**
   * @brief Return the static number of outputs or calculate regular outputs using output_fn
   */
  DLL_PUBLIC int CalculateOutputs(const OpSpec &spec) const;

  /**
   * @brief Calculate the number of additional outputs obtained from additional_outputs_fn
   */
  DLL_PUBLIC int CalculateAdditionalOutputs(const OpSpec &spec) const;

  DLL_PUBLIC bool SupportsInPlace(const OpSpec &spec) const;

  DLL_PUBLIC void CheckArgs(const OpSpec &spec) const;

  /**
   * @brief Get default value of optional or internal argument. The default value must be declared
   */
  template <typename T>
  DLL_PUBLIC inline T GetDefaultValueForArgument(const std::string &s) const;

  DLL_PUBLIC bool HasRequiredArgument(const std::string &name, bool local_only = false) const;

  DLL_PUBLIC bool HasOptionalArgument(const std::string &name, bool local_only = false) const;

  DLL_PUBLIC bool HasInternalArgument(const std::string &name, bool local_only = false) const;

  /**
   * @brief Finds default value for a given argument
   * @return A pair of the defining schema and the value
   */
  DLL_PUBLIC std::pair<const OpSchema *, const Value *> FindDefaultValue(
      const std::string &arg_name, bool local_only = false, bool include_internal = true) const;

  /**
   * @brief Checks whether the schema defines an argument with the given name
   * @param include_internal - returns `true` also for internal/implicit arugments
   * @param local_only       - doesn't look in parent schemas
   */
  DLL_PUBLIC bool HasArgument(const std::string &name,
                              bool include_internal = false,
                              bool local_only = false) const;

  /**
   * @brief Get docstring for operator argument of given name (Python Operator Kwargs).
   */
  DLL_PUBLIC std::string GetArgumentDox(const std::string &name) const;

  /**
   * @brief Get enum representing type of argument of given name.
   */
  DLL_PUBLIC DALIDataType GetArgumentType(const std::string &name) const;

  /**
   * @brief Check if the argument has a default value.
   *        Required arguments always return false.
   *        Internal arguments always return true.
   */
  DLL_PUBLIC bool HasArgumentDefaultValue(const std::string &name) const;

  /**
   * @brief Get default value of optional argument represented as python-compatible repr string.
   *        Not allowed for internal arguments.
   */
  DLL_PUBLIC std::string GetArgumentDefaultValueString(const std::string &name) const;

  /**
   * @brief Get names of all required, optional, and deprecated arguments
   */
  DLL_PUBLIC std::vector<std::string> GetArgumentNames() const;
  DLL_PUBLIC bool IsTensorArgument(const std::string &name) const;
  DLL_PUBLIC bool ArgSupportsPerFrameInput(const std::string &arg_name) const;

 private:
  static inline bool ShouldHideArgument(const std::string &name) {
    return name.size() && name[0] == '_';
  }

  const TensorArgDesc *FindTensorArgument(const std::string &name) const;

  void CheckArgument(const std::string &s);

  void CheckInputIndex(int index) const;

  std::string DefaultDeprecatedArgMsg(const std::string &arg_name, const std::string &renamed_to,
                                      bool removed) const;

  /**
   * @brief Add internal argument to schema. It always has a value.
   */
  template <typename T>
  void AddInternalArg(const std::string &name, const std::string &doc, T value) {
    auto v = Value::construct(value);
    internal_arguments_[name] = {doc, type2id<T>::value, v.get(), true};
    internal_arguments_unq_.push_back(std::move(v));
  }

  std::map<std::string, RequiredArgumentDef> GetRequiredArguments() const;
  std::map<std::string, DefaultedArgumentDef> GetOptionalArguments() const;
  std::map<std::string, DeprecatedArgDef> GetDeprecatedArguments() const;

  string dox_;
  string name_;

  bool disable_auto_input_dox_ = false;

  struct InputDoc {
    std::string name = {};
    std::string type_doc = {};
    std::string doc = {};
  };
  std::vector<InputDoc> input_dox_ = {};
  bool input_dox_set_ = false;

  // Custom docstring, if not empty should be used in place of input_dox_ descriptions
  std::string call_dox_ = {};

  // Whether to append kwargs section to __call__ docstring. Off by default,
  // can be turned on for call_dox_ specified manually
  bool append_kwargs_section_ = false;

  SpecFunc output_fn_, in_place_fn_, additional_outputs_fn_;

  int min_num_input_ = 0, max_num_input_ = 0;
  int num_output_ = 0;

  bool allow_instance_grouping_ = true;
  vector<string> parents_;

  bool support_volumetric_ = false;

  bool allow_sequences_ = false;
  bool is_sequence_operator_ = false;

  bool is_internal_ = false;
  bool is_doc_hidden_ = false;
  bool is_doc_partially_hidden_ = false;

  bool no_prune_ = false;

  bool serializable_ = true;

  std::map<int, int> passthrough_map_;
  bool samplewise_any_passthrough_ = false;

  bool is_deprecated_ = false;
  std::string deprecated_in_favor_of_;
  std::string deprecation_message_;

  std::map<std::string, RequiredArgumentDef> arguments_;
  std::map<std::string, DefaultedArgumentDef> optional_arguments_;
  std::map<std::string, DefaultedArgumentDef> internal_arguments_;
  std::map<std::string, DeprecatedArgDef> deprecated_arguments_;
  std::vector<std::unique_ptr<Value>> optional_arguments_unq_;
  std::vector<std::unique_ptr<Value>> internal_arguments_unq_;
  std::vector<std::vector<TensorLayout>> input_layouts_;
  std::vector<dali::InputDevice> input_devices_;

  std::map<std::string, TensorArgDesc> tensor_arguments_;
};

class SchemaRegistry {
 public:
  DLL_PUBLIC static OpSchema &RegisterSchema(const std::string &name);
  DLL_PUBLIC static const OpSchema &GetSchema(const std::string &name);
  DLL_PUBLIC static const OpSchema *TryGetSchema(const std::string &name);

 private:
  inline SchemaRegistry() {}

  DLL_PUBLIC static std::map<string, OpSchema> &registry();
};

template <typename T>
inline T OpSchema::GetDefaultValueForArgument(const std::string &s) const {
  const Value *v = FindDefaultValue(s, false, true).second;
  DALI_ENFORCE(v != nullptr,
               make_string("The argument \"", s, "\" doesn't have a default value in schema \"",
                           name(), "\"."));

  using S = argument_storage_t<T>;
  const ValueInst<S> *vS = dynamic_cast<const ValueInst<S> *>(v);
  DALI_ENFORCE(vS != nullptr, "Unexpected type of the default value for argument \"" + s +
                                  "\" of schema \"" + this->name() + "\"");
  return static_cast<T>(vS->Get());
}

#define DALI_SCHEMA_REG(OpName)                         \
  int DALI_OPERATOR_SCHEMA_REQUIRED_FOR_##OpName() {    \
    return 42;                                          \
  }                                                     \
  static ::dali::OpSchema *ANONYMIZE_VARIABLE(OpName) = \
      &::dali::SchemaRegistry::RegisterSchema(#OpName)

#define DALI_SCHEMA(OpName) DALI_SCHEMA_REG(OpName)

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_OP_SCHEMA_H_
