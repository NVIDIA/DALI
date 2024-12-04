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

#ifndef DALI_PIPELINE_OPERATOR_OP_SCHEMA_H_
#define DALI_PIPELINE_OPERATOR_OP_SCHEMA_H_

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
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

enum class InputDevice : uint8_t {
  /** CPU for CPU and Mixed operators; GPU for GPU operators. */
  MatchBackend = 0,

  /** Always CPU */
  CPU,

  /** Always GPU */
  GPU,

  /** Any kind of input device, regardless of operator's backend */
  Any,

  /** CPU for CPU and Mixed; anything for GPU */
  MatchBackendOrCPU,

  /** Any backend, but the operator will not access the actual data.
   *
   * This is useful for operators that only look at the metadata of the input - things like shape,
   * source info, dtype, etc.
   *
   * Specifying this flag allows the executor to skip synchronization or even to provide
   * a tensor without actual payload.
   * It is forbidden to:
   * - look at the data inside the operator (`data`, `raw_data`, etc)
   * - copy the input data (`TensorList::Copy` or calling copy on the samples)
   * - forward the input or its parts to the output with ShareData, SetSample or similar.
   */
  Metadata,
};

struct ArgumentDeprecation {
  std::string renamed_to = {};
  std::string msg = {};
  bool removed = false;
};

struct ArgumentDef {
  std::string name;
  std::string doc;
  DALIDataType dtype;
  std::unique_ptr<Value> default_value;
  bool required   : 1 = false;
  bool tensor     : 1 = false;
  bool per_frame  : 1 = false;
  bool internal   : 1 = false;
  bool hidden     : 1 = false;

  std::unique_ptr<ArgumentDeprecation> deprecated;
};

struct InputInfo {
  std::string name;
  InputDevice device;

  struct InputDoc {
    std::string type_doc;
    std::string doc;
  };

  std::vector<TensorLayout> layouts;
};


class DLL_PUBLIC OpSchema {
 public:
  typedef std::function<int(const OpSpec &spec)> SpecFunc;

  OpSchema(OpSchema &&) = default;
  OpSchema(const OpSchema &) = delete;
  OpSchema &operator=(const OpSchema &) = delete;
  OpSchema &operator=(OpSchema &&) = default;

  DLL_PUBLIC explicit OpSchema(std::string_view name);

  DLL_PUBLIC inline ~OpSchema() = default;

  /**  Returns an empty schema, with only internal arguments */
  DLL_PUBLIC static const OpSchema &Default();

  /**  Returns the schema name of this operator. */
  DLL_PUBLIC std::string_view name() const;

  /** Returns the module path of this operator. */
  DLL_PUBLIC const std::vector<std::string> &ModulePath() const;

  /** Returns the camel case name of the operator (without the module path) */
  DLL_PUBLIC std::string_view OperatorName() const;

  /** Sets the doc string for this operator. */
  DLL_PUBLIC OpSchema &DocStr(std::string_view dox);

  /** Sets the docstring for input.
   *
   * Set the documentation for intput at given `index`.
   *
   * If the operator specifies some range of allowed inputs with NumInput(int min, int max)
   * only the first `min` inputs are considered mandatory, the rest are optional
   *
   * Will generate entry in `Args` section using numpydoc style:
   * name : type_doc
   *     doc
   */
  DLL_PUBLIC OpSchema &InputDox(int index, std::string_view name, std::string_view type_doc,
                                std::string_view doc);

  /** Allows to set a docstring for __call__ method of Operator.
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
   * input0 : Type of input
   *     This is the first input
   * input1 : TensorList of some kind
   *     This is second input
   * optional_input : TensorList, optional
   *     This is optional input
   *
   * If the `append_kwargs_section` is true, the docstring generator will append the Keyword args
   * section at the end of this doc
   *
   * @param doc
   * @param append_kwargs_section
   */
  DLL_PUBLIC OpSchema &CallDocStr(std::string_view doc, bool append_kwargs_section = false);

  /** Sets a function that infers the number of outputs this op will produce from OpSpec.
   *
   * This is required to expose the op to the python interface.
   *
   * If the ops has a fixed number of outputs, this function
   * does not need to be added to the schema.
   */
  DLL_PUBLIC OpSchema &OutputFn(SpecFunc f);

  /**  Sets a function to determine the number of additional outputs from the OpSpec.
   *
   * If this function is not set it will be assumed that no
   * additional outputs can be returned
   *
   * Use case is to expose additional information (such as random
   * numbers used within operators) to the user
   */
  DLL_PUBLIC OpSchema &AdditionalOutputsFn(SpecFunc f);

  /** Sets the number of inputs that the op can receive. */
  DLL_PUBLIC OpSchema &NumInput(int n);

  /** Sets the min and max number of inputs the op can receive. */
  DLL_PUBLIC OpSchema &NumInput(int min, int max);

  /** Sets the input device for given range of inputs */
  DLL_PUBLIC OpSchema &InputDevice(int first, int one_past, dali::InputDevice device);

  /** Sets the input device for given range of input */
  DLL_PUBLIC OpSchema &InputDevice(int index, dali::InputDevice device);

  /** Gets the supported input device for given input */
  DLL_PUBLIC dali::InputDevice GetInputDevice(int index) const;

  /** Sets the number of outputs that the op can receive. */
  DLL_PUBLIC OpSchema &NumOutput(int n);

  /**
   * @brief Indicates that this operator should not use auto-generated documentation
   *        of inputs and `__call__` operator with custom signature.
   */
  DLL_PUBLIC OpSchema &DisableAutoInputDox();

  /**
   * @brief Indicates that multiple instances of this operator cannot share a logical ID to achieve
   *        uniform processing of multiple input sets
   */
  DLL_PUBLIC OpSchema &DisallowInstanceGrouping();

  /** Notes that this operator expects sequence inputs exclusively */
  DLL_PUBLIC OpSchema &SequenceOperator();

  /** Notes that sequences can be used with this op */
  DLL_PUBLIC OpSchema &AllowSequences();

  /** Notes that the operator can process 3D data. */
  DLL_PUBLIC OpSchema &SupportVolumetric();

  /** Notes that this operator is internal and shouldn't be exposed in Python API. */
  DLL_PUBLIC OpSchema &MakeInternal();

  /** Notes that this operator doc should not be visible (but the Op is exposed in Python API) */
  DLL_PUBLIC OpSchema &MakeDocHidden();

  /**
   * @brief Notes that for this operator only the doc_str should be visible, but not the docs for
   *        the inputs, outputs or argument (the Op is exposed in Python API)
   */
  DLL_PUBLIC OpSchema &MakeDocPartiallyHidden();

  /**  Notes that this operator is deprecated and optionally specifies its successor
   *
   * @param in_favor_of schema name of the replacement
   * @param explanation additional explanation
   */
  DLL_PUBLIC OpSchema &Deprecate(std::string_view in_favor_of = "",
                                 std::string_view explanation = "");

  /** Notes that this operator cannot be serialized */
  DLL_PUBLIC OpSchema &Unserializable();

  /** Adds a required argument to op with its type */
  DLL_PUBLIC OpSchema &AddArg(std::string_view s, std::string_view doc,
                              const DALIDataType dtype, bool enable_tensor_input = false,
                              bool support_per_frame_input = false);


  /** Adds a required argument of type DALIDataType */
  DLL_PUBLIC OpSchema &AddTypeArg(std::string_view s, std::string_view doc);

  /** Sets input layout constraints and default for given input.
   *
   * At run-time, when the operator encounters a tensor(list) with specified
   * layout, but different than one provided to this function, error is raised.
   *
   * If the input tensor has no layout, the one provided to this function is assumed
   * if number of dimensions matches. Otherswise, error is raised.
   */
  DLL_PUBLIC OpSchema &InputLayout(int index, TensorLayout layout);

  /** Sets input layout constraints and default for given input.
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

  /** Sets input layout constraint and default for all inputs.
   *
   * @see InputLayout(int index, TensorLayout layout)
   */
  DLL_PUBLIC OpSchema &InputLayout(TensorLayout layout);

  /**  Sets input layout constraint and default for all inputs.
   *
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
  DLL_PUBLIC OpSchema &AddOptionalArg(std::string_view s, std::string_view doc,
                                      DALIDataType dtype, std::nullptr_t,
                                      bool enable_tensor_input = false,
                                      bool support_per_frame_input = false);

  /**
   * @brief Adds an optional non-vector argument without default to op.
   *        If the arg name starts is with an underscore, it will be marked hidden, which
   *        makes it not listed in the docs.
   */
  template <typename T>
  DLL_PUBLIC inline OpSchema &AddOptionalArg(std::string_view s, std::string_view doc,
                                             std::nullptr_t, bool enable_tensor_input = false,
                                             bool support_per_frame_input = false) {
    AddOptionalArg(s, doc, type2id<T>::value, nullptr, enable_tensor_input,
                   support_per_frame_input);
    return *this;
  }


  /**  Adds an optional non-vector argument to op
   *
   * If the arg name starts is with an underscore, it will be marked hidden, which
   * makes it not listed in the docs.
   */
  template <typename T>
  DLL_PUBLIC inline std::enable_if_t<!is_vector<T>::value && !is_std_array<T>::value, OpSchema &>
  AddOptionalArg(std::string_view s, std::string_view doc, T default_value,
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

  /** Adds an optional argument of type DALIDataType with a default value
   *
   * If the arg name starts is with an underscore, it will be marked hidden, which
   * makes it not listed in the docs.
   */
  DLL_PUBLIC OpSchema &AddOptionalTypeArg(std::string_view s, std::string_view doc,
                                          DALIDataType default_value);

  /**  Adds an optional argument of type DALIDataType without a default value
   *
   * If the arg name starts is with an underscore, it will be marked hidden, which
   * makes it not listed in the docs.
   */
  DLL_PUBLIC OpSchema &AddOptionalTypeArg(std::string_view s, std::string_view doc);

  DLL_PUBLIC OpSchema &AddOptionalArg(std::string_view s, std::string_view doc,
                                      const char *default_value);

  /**  Adds an optional vector argument to op
   *
   * If the arg name starts is with an underscore, it will be marked hidden, which
   * makes it not listed in the docs.
   */
  template <typename T>
  DLL_PUBLIC inline OpSchema &AddOptionalArg(std::string_view s, std::string_view doc,
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

  DLL_PUBLIC OpSchema &AddRandomSeedArg();

  /**  Marks an argument as deprecated in favor of a new argument
   *
   * Providing renamed_to means the argument has been renamed and we can safely
   * propagate the value to the new argument name.
   */
  DLL_PUBLIC OpSchema &DeprecateArgInFavorOf(std::string_view arg_name, std::string renamed_to,
                                             std::string msg = {});

  /**  Marks an argument as deprecated
   *
   * @remarks There are three ways to deprecate an argument
   *          1. removed==true, means the operator will not use the
   *              argument at all and it can be safely discarded.
   *          2. removed==false, means the operator will still use the
   *              deprecated argument until it is finally removed completely from the schema.
   *          3. For renaming the argument see DeprecateArgInFavorOf
   */
  DLL_PUBLIC OpSchema &DeprecateArg(std::string_view arg_name, bool removed = true,
                                    std::string msg = {});

  /**
   * @brief Sets a function that infers whether the op can
   *        be executed in-place depending on the ops specification.
   */
  DLL_PUBLIC OpSchema &InPlaceFn(SpecFunc f);

  /** Sets a parent (which could be used as a storage of default parameters)
   *
   * Does not support cyclic dependency. There can be multiple parents and the lookup is transitive.
   * Only arguments are inherited, inputs and outputs are not.
   */
  DLL_PUBLIC OpSchema &AddParent(std::string_view parentName);

  /**  Notes that this operator should not be pruned from a graph even if its outputs are unused. */
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
   *
   * The data is passed on sample basis, allowing to mix any input to any output.
   */
  DLL_PUBLIC OpSchema &SamplewisePassThrough();

  /** Get parent schemas (non-recursive) */
  DLL_PUBLIC const vector<std::string> &GetParentNames() const;

  DLL_PUBLIC const vector<const OpSchema *> &GetParents() const;

  /** Get the docstring of the operator - provided by DocStr in the schema definition. */
  DLL_PUBLIC string Dox() const;

  /** Return true wether the default input docs can be used */
  DLL_PUBLIC bool CanUseAutoInputDox() const;

  /**
   * @brief Whether the docstring for kwargs should be automatically generated and appended to the
   *        one provided in CallDocStr.
   */
  DLL_PUBLIC bool AppendKwargsSection() const;

  /**  Return true when `__call__` docstring was explicitly set
   *
   * Should be considered as highest preference
   */
  DLL_PUBLIC bool HasCallDox() const;

  /** Get the documentation for Operator __call__ signature provided by CallDocStr. */
  DLL_PUBLIC std::string GetCallDox() const;

  /** Check if this operator has input docstrings provided */
  DLL_PUBLIC bool HasInputDox() const;

  /**
   * @brief List all the inputs that should appear in `__call__` signature based on the input
   *        docs that were specified. Requires HasInputDox() to return true
   *
   */
  DLL_PUBLIC std::string GetCallSignatureInputs() const;

  /** Get the docstring name of the input at given index. */
  DLL_PUBLIC std::string GetInputName(int input_idx) const;

  /** Get the docstring type of the input at given index. */
  DLL_PUBLIC std::string GetInputType(int input_idx) const;

  /** Get the docstring text of the input at given index. */
  DLL_PUBLIC std::string GetInputDox(int input_idx) const;

  /** Get the maximal number of accepted inputs. */
  DLL_PUBLIC int MaxNumInput() const;

  /** Get the minimal number of required inputs. */
  DLL_PUBLIC int MinNumInput() const;

  /** Get the number of static outputs, see also CalculateOutputs and CalculateAdditionalOutputs */
  DLL_PUBLIC int NumOutput() const;

  DLL_PUBLIC bool AllowsInstanceGrouping() const;

  /** Whether this operator accepts ONLY sequences as inputs */
  DLL_PUBLIC bool IsSequenceOperator() const;

  /** Whether this operator accepts sequences as inputs */
  DLL_PUBLIC bool AllowsSequences() const;

  /** Whether this operator accepts volumes as inputs */
  DLL_PUBLIC bool SupportsVolumetric() const;

  /**  Whether this operator is internal to DALI backend (and shouldn't be exposed in Python API) */
  DLL_PUBLIC bool IsInternal() const;

  /** Whether this operator doc should not be visible (but the Op is exposed in Python API) */
  DLL_PUBLIC bool IsDocHidden() const;

  /**
   * Whether this operator doc should be visible without documenting any parameters.
   * Useful for deprecated ops.
   */
  DLL_PUBLIC bool IsDocPartiallyHidden() const;

  /** Whether this operator is deprecated. */
  DLL_PUBLIC bool IsDeprecated() const;

  /** What operator replaced the current one. */
  DLL_PUBLIC std::string_view DeprecatedInFavorOf() const;

  /** Additional deprecation message */
  DLL_PUBLIC std::string_view DeprecationMessage() const;

  /** Whether given argument is deprecated. */
  DLL_PUBLIC bool IsDeprecatedArg(std::string_view arg_name) const;

  /** Information about the argument deprecation - error message, renaming, removal, etc. */
  DLL_PUBLIC const ArgumentDeprecation &DeprecatedArgInfo(std::string_view arg_name) const;

  /** Check whether this operator calculates number of outputs statically
   *
   * @return false if static, true if dynamic
   */
  DLL_PUBLIC bool HasOutputFn() const;

  /** Check whether this operator won't be pruned out of graph even if not used. */
  DLL_PUBLIC bool IsNoPrune() const;

  DLL_PUBLIC bool IsSerializable() const;

  /**  Returns the index of the output to which the input is passed.
   *
   * @param strict consider only fully passed through batches
   * @return Output indicies or empty vector if given input is not passed through.
   */
  DLL_PUBLIC std::vector<int> GetPassThroughOutputIdx(int input_idx, const OpSpec &spec,
                                                      bool strict = true) const;

  /** Is the input_idx passed through to output_idx */
  DLL_PUBLIC bool IsPassThrough(int input_idx, int output_idx, bool strict = true) const;

  /** Does this operator pass through any data? */
  DLL_PUBLIC bool HasPassThrough() const;

  /** Does this operator pass through any data as a whole batch to batch? */
  DLL_PUBLIC bool HasStrictPassThrough() const;

  /** Does this operator pass through any data by the means of sharing individual samples? */
  DLL_PUBLIC bool HasSamplewisePassThrough() const;

  /** Return the static number of outputs or calculate regular outputs using output_fn */
  DLL_PUBLIC int CalculateOutputs(const OpSpec &spec) const;

  /** Calculate the number of additional outputs obtained from additional_outputs_fn */
  DLL_PUBLIC int CalculateAdditionalOutputs(const OpSpec &spec) const;

  DLL_PUBLIC bool SupportsInPlace(const OpSpec &spec) const;

  DLL_PUBLIC void CheckArgs(const OpSpec &spec) const;

  /** Get default value of optional or internal argument. The default value must be declared */
  template <typename T>
  DLL_PUBLIC inline T GetDefaultValueForArgument(std::string_view s) const;

  DLL_PUBLIC bool HasRequiredArgument(std::string_view name, bool local_only = false) const;

  DLL_PUBLIC bool HasOptionalArgument(std::string_view name, bool local_only = false) const;

  DLL_PUBLIC bool HasInternalArgument(std::string_view name, bool local_only = false) const;

  /** Finds default value for a given argument
   *
   * @return A pair of the defining schema and the value
   */
  DLL_PUBLIC std::pair<const OpSchema *, const Value *> FindDefaultValue(
      std::string_view arg_name, bool local_only = false, bool include_internal = true) const;

  /** Checks whether the schema defines an argument with the given name
   *
   * @param include_internal - returns `true` also for internal/implicit arugments
   * @param local_only       - doesn't look in parent schemas
   */
  DLL_PUBLIC bool HasArgument(std::string_view name,
                              bool include_internal = false,
                              bool local_only = false) const;

  /** Returns true if the operator has a "seed" argument. */
  DLL_PUBLIC bool HasRandomSeedArg() const;

  /** Get docstring for operator argument of given name (Python Operator Kwargs). */
  DLL_PUBLIC std::string GetArgumentDox(std::string_view name) const;

  /** Get enum representing type of argument of given name. */
  DLL_PUBLIC DALIDataType GetArgumentType(std::string_view name) const;

  /** Check if the argument has a default value.
   *
   * Required arguments always return false.
   * Internal arguments always return true.
   */
  DLL_PUBLIC bool HasArgumentDefaultValue(std::string_view name) const;

  /**
   * @brief Get default value of optional argument represented as python-compatible repr string.
   *        Not allowed for internal arguments.
   */
  DLL_PUBLIC std::string GetArgumentDefaultValueString(std::string_view name) const;

  /** Get names of all required, optional, and deprecated arguments */
  DLL_PUBLIC std::vector<std::string> GetArgumentNames() const;
  DLL_PUBLIC bool IsTensorArgument(std::string_view name) const;
  DLL_PUBLIC bool ArgSupportsPerFrameInput(std::string_view arg_name) const;

 private:
  static inline bool ShouldHideArgument(std::string_view name) {
    return name.size() && name[0] == '_';
  }

  const ArgumentDef *FindTensorArgument(std::string_view name) const;

  template <typename Pred>
  const ArgumentDef *FindArgument(std::string_view name, Pred &&pred) {
      auto it = arguments_.find(name);
    if (it != tensor_arguments_.end()) {
      return pred(it->second) ? &it->second : nullptr;
    }
    for (const OpSchema *parent : GetParents()) {
      auto desc = parent->FindTensorArgument(name);
      if (desc) {
        return desc;
      }
    }
    return nullptr;
  }


  void CheckArgument(std::string_view s);

  void CheckInputIndex(int index) const;

  std::string DefaultDeprecatedArgMsg(std::string_view arg_name, std::string_view renamed_to,
                                      bool removed) const;

  /** Add internal argument to schema. It always has a value. */
  template <typename T>
  void AddInternalArg(std::string_view name, std::string_view doc, T value) {
    auto &arg = AddArgumentImpl(name, type2id<T>::value, Value::construct(value), doc);
    arg.hidden = true;
    arg.internal = true;
  }

  DLL_PUBLIC ArgumentDef &AddArgumentImpl(std::string_view name,
                                          DALIDataType type,
                                          std::unique_ptr<Value> default_value,
                                          std::string doc);

  /** Initialize the module_path_ and operator_name_ fields based on the schema name. */
  void InitNames();

  /** Populates the default schema with common arguments */
  void InitDefaultSchema();

  std::string dox_;
  /// The name of the schema
  std::string name_;
  /// The module path for the operator
  std::vector<std::string> module_path_;
  /// The PascalCase name of the operator (without the module path)
  std::string operator_name_;

  ////////////////////////////////////////////////////////////////////////////
  // Inputs, outputs and arguments

  /// All arguments
  std::map<std::string, ArgumentDef, std::less<>> arguments_;

  /// The properties of the inputs
  std::vector<InputInfo> input_info_;
  bool disable_auto_input_dox_ = false;
  bool input_dox_set_ = false;

  SpecFunc output_fn_, in_place_fn_, additional_outputs_fn_;

  int min_num_input_ = 0, max_num_input_ = 0;
  int num_output_ = 0;

  ////////////////////////////////////////////////////////////////////////////
  // Schema inheritance

  /// Names of the parent schemas
  vector<string> parent_names_;
  /// Cached pointers to parent schemas, to avoid repeated lookups
  mutable std::vector<const OpSchema *> parents_;
  mutable std::mutex parents_lock_;

  ////////////////////////////////////////////////////////////////////////////
  // Documentation-related
  bool support_volumetric_ = false;
  bool allow_sequences_ = false;
  bool is_sequence_operator_ = false;

  bool is_internal_ = false;
  bool is_doc_hidden_ = false;
  bool is_doc_partially_hidden_ = false;

  /// Custom docstring, if not empty should be used in place of input_dox_ descriptions
  std::string call_dox_ = {};

  /// Whether to append kwargs section to __call__ docstring. Off by default,
  /// can be turned on for call_dox_ specified manually
  bool append_kwargs_section_ = false;

  ////////////////////////////////////////////////////////////////////////////
  // Internal flags
  bool allow_instance_grouping_ = true;
  bool no_prune_ = false;
  bool serializable_ = true;

  ////////////////////////////////////////////////////////////////////////////
  // Passthrough operators
  std::map<int, int> passthrough_map_;
  bool samplewise_any_passthrough_ = false;

  ////////////////////////////////////////////////////////////////////////////
  // Deprecation
  bool is_deprecated_ = false;
  std::string deprecated_in_favor_of_;
  std::string deprecation_message_;
};

class SchemaRegistry {
 public:
  DLL_PUBLIC static OpSchema &RegisterSchema(std::string_view name);
  DLL_PUBLIC static const OpSchema &GetSchema(std::string_view name);
  DLL_PUBLIC static const OpSchema *TryGetSchema(std::string_view name);

 private:
  inline SchemaRegistry() {}

  DLL_PUBLIC static std::map<string, OpSchema, std::less<>> &registry();
};

template <typename T>
inline T OpSchema::GetDefaultValueForArgument(std::string_view s) const {
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
