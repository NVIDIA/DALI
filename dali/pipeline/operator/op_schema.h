// Copyright (c) 2017-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <optional>
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
  ArgumentDeprecation() = default;
  ArgumentDeprecation(string renamed_to, string msg, bool removed = false)
  : renamed_to(renamed_to), msg(msg), removed(removed) {}

  std::string renamed_to;
  std::string msg;
  bool removed = false;
};

class OpSchema;

struct ArgumentDef {
  const OpSchema *defined_in;
  std::string name;
  std::string doc;
  DALIDataType dtype;

  // TODO(michalz): Convert to bit fields in C++20 (before C++20 bit fields can't have initializers)
  bool required   = false;  //< The argument must be set.
  bool tensor     = false;  //< The argument can be provided as a TensorList
  bool per_frame  = false;  //< The (tensor) argument can be expanded to multiple frames
  bool internal   = false;  //< The argument cannot be set by the user in Python
  bool hidden     = false;  //< The argument doesn't appear in the documentation
  bool ignore_cmp = false;  //< Two operators can be considered equal if this argument differs

  std::unique_ptr<Value> default_value;
  std::unique_ptr<ArgumentDeprecation> deprecated;
};

struct InputInfo {
  std::string name;
  InputDevice device;

  struct InputDoc {
    std::string type_doc;
    std::string doc;
  } doc;

  std::vector<TensorLayout> layouts;
};

namespace detail {

/** A helper class for lazy evaluation
 *
 * In some cases, it's impossible or wasteful to compute a value eagerly.
 * This class provides a thread-safe storage for such value.
 *
 * Usage: place in your class (possibly as a mutable field) and call Get with a function
 * that returns a value convertible to T. This function will be called only once per LazyValue's
 * lifetime.
 *
 * NOTE: Copying the lazy value is a no-op.
 */
template <typename T>
struct LazyValue {
  LazyValue() = default;
  LazyValue(const LazyValue &) {}
  LazyValue(LazyValue &&) {}
  LazyValue &operator=(const LazyValue &) { return *this; }
  LazyValue &operator=(LazyValue &&) { return *this; }

  template <typename PopulateFn>
  T &Get(PopulateFn &&fn) {
    if (data)
      return *data;
    std::lock_guard g(lock);
    if (data)
      return *data;
    data = std::make_unique<T>(fn());
    return *data;
  }
  std::unique_ptr<T> data;
  std::recursive_mutex lock;
};
}  // namespace detail


class DLL_PUBLIC OpSchema {
 public:
  typedef std::function<int(const OpSpec &spec)> SpecFunc;

  OpSchema(OpSchema &&) = delete;
  OpSchema(const OpSchema &) = delete;
  OpSchema &operator=(const OpSchema &) = delete;
  OpSchema &operator=(OpSchema &&) = delete;

  explicit OpSchema(std::string_view name);

  inline ~OpSchema() = default;

  /**  Returns an empty schema, with only internal arguments */
  static const OpSchema &Default();

  /**  Returns the schema name of this operator. */
  const std::string &name() const;

  /** Returns the module path of this operator. */
  const std::vector<std::string> &ModulePath() const;

  /** Returns the camel case name of the operator (without the module path) */
  const std::string &OperatorName() const;

  /** Sets the doc string for this operator. */
  OpSchema &DocStr(std::string dox);

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
  OpSchema &InputDox(int index, std::string_view name, std::string type_doc, std::string doc);

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
  OpSchema &CallDocStr(std::string doc, bool append_kwargs_section = false);

  /** Sets a function that infers the number of outputs this op will produce from OpSpec.
   *
   * This is required to expose the op to the python interface.
   *
   * If the ops has a fixed number of outputs, this function
   * does not need to be added to the schema.
   */
  OpSchema &OutputFn(SpecFunc f);

  /**  Sets a function to determine the number of additional outputs from the OpSpec.
   *
   * If this function is not set it will be assumed that no
   * additional outputs can be returned
   *
   * Use case is to expose additional information (such as random
   * numbers used within operators) to the user
   */
  OpSchema &AdditionalOutputsFn(SpecFunc f);

  /** Sets the number of inputs that the op can receive. */
  OpSchema &NumInput(int n);

  /** Sets the min and max number of inputs the op can receive. */
  OpSchema &NumInput(int min, int max);

  /** Sets the input device for given range of inputs */
  OpSchema &InputDevice(int first, int one_past, dali::InputDevice device);

  /** Sets the input device for given range of input */
  OpSchema &InputDevice(int index, dali::InputDevice device);

  /** Gets the supported input device for given input */
  dali::InputDevice GetInputDevice(int index) const;

  /** Sets the number of outputs that the op can receive. */
  OpSchema &NumOutput(int n);

  /**
   * @brief Indicates that this operator should not use auto-generated documentation
   *        of inputs and `__call__` operator with custom signature.
   */
  OpSchema &DisableAutoInputDox();

  /** Notes that this operator expects sequence inputs exclusively */
  OpSchema &SequenceOperator();

  /** Notes that sequences can be used with this op */
  OpSchema &AllowSequences();

  /** Notes that the operator can process 3D data. */
  OpSchema &SupportVolumetric();

  /** Notes that this operator is internal and shouldn't be exposed in Python API. */
  OpSchema &MakeInternal();

  /** Notes that this operator doc should not be visible (but the Op is exposed in Python API) */
  OpSchema &MakeDocHidden();

  /** Notes that this operator doesn't have a state.
   *
   * NOTE: This overrides the statefulness inherited from parent schemas.
   */
  OpSchema &MakeStateless();

  /** Notes that this operator is stateful. */
  OpSchema &MakeStateful();

  /**
   * @brief Notes that for this operator only the doc_str should be visible, but not the docs for
   *        the inputs, outputs or argument (the Op is exposed in Python API)
   */
  OpSchema &MakeDocPartiallyHidden();

  /**  Notes that this operator is deprecated and optionally specifies its successor
   *
   * @param in_favor_of schema name of the replacement
   * @param explanation additional explanation
   */
  OpSchema &Deprecate(std::string in_favor_of = "",
                      std::string explanation = "");

  /** Notes that this operator cannot be serialized */
  OpSchema &Unserializable();

  /** Adds a required argument to op with its type */
  OpSchema &AddArg(std::string_view s, std::string doc,
                   const DALIDataType dtype, bool enable_tensor_input = false,
                   bool support_per_frame_input = false);


  /** Adds a required argument of type DALIDataType */
  OpSchema &AddTypeArg(std::string_view s, std::string doc);

  /** Sets input layout constraints and default for given input.
   *
   * At run-time, when the operator encounters a tensor(list) with specified
   * layout, but different than one provided to this function, error is raised.
   *
   * If the input tensor has no layout, the one provided to this function is assumed
   * if number of dimensions matches. Otherswise, error is raised.
   */
  OpSchema &InputLayout(int index, TensorLayout layout);

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
  OpSchema &InputLayout(int index, std::initializer_list<TensorLayout> layouts);

  /** Sets input layout constraint and default for all inputs.
   *
   * @see InputLayout(int index, TensorLayout layout)
   */
  OpSchema &InputLayout(TensorLayout layout);

  /**  Sets input layout constraint and default for all inputs.
   *
   * @see InputLayout(int index, TensorLayout layout)
   */
  OpSchema &InputLayout(std::initializer_list<TensorLayout> layouts);

  /**
   * @brief Verifies that the layout is valid for given input index and number of dimensions
   *        or returns a default layout if the layout parameter is empty.
   */
  const TensorLayout &GetInputLayout(int index, int sample_ndim,
                                     const TensorLayout &layout = {}) const;


  const std::vector<TensorLayout> &GetSupportedLayouts(int input_idx) const;

  /**
   * @brief Adds an optional non-vector argument without default to op
   *        The type can be specified as enum, nullptr_t is used for overload resolution
   *        If the arg name starts with an underscore, it will be marked hidden, which
   *        makes it not listed in the docs.
   */
  OpSchema &AddOptionalArg(std::string_view s, std::string doc,
                           DALIDataType dtype, std::nullptr_t,
                           bool enable_tensor_input = false,
                           bool support_per_frame_input = false);

  /**
   * @brief Adds an optional non-vector argument without default to op.
   *        If the arg name starts with an underscore, it will be marked hidden, which
   *        makes it not listed in the docs.
   */
  template <typename T>
  inline OpSchema &AddOptionalArg(std::string_view name, std::string doc,
                                  std::nullptr_t, bool enable_tensor_input = false,
                                  bool support_per_frame_input = false) {
    AddOptionalArg(name, doc, type2id<T>::value, nullptr, enable_tensor_input,
                   support_per_frame_input);
    return *this;
  }


  /**  Adds an optional non-vector argument to op
   *
   * If the arg name starts with an underscore, it will be marked hidden, which
   * makes it not listed in the docs.
   */
  template <typename T>
  inline std::enable_if_t<!is_vector<T>::value && !is_std_array<T>::value, OpSchema &>
  AddOptionalArg(std::string_view name, std::string doc, T default_value,
                 bool enable_tensor_input = false, bool support_per_frame_input = false) {
    static_assert(
        !std::is_same<T, DALIDataType>::value,
        R"(Use `AddOptionalTypeArg` instead. `AddOptionalArg` with a default value should not be
used with DALIDataType, to avoid confusion with `AddOptionalArg<type>(name, doc, nullptr)`)");
    auto &arg = AddArgumentImpl(name, type2id<T>::value,
                                Value::construct(default_value),
                                std::move(doc));
    arg.tensor = enable_tensor_input;
    arg.per_frame = support_per_frame_input;
    return *this;
  }

  /** Adds an optional argument of type DALIDataType with a default value
   *
   * If the arg name starts with an underscore, it will be marked hidden, which
   * makes it not listed in the docs.
   */
  OpSchema &AddOptionalTypeArg(std::string_view name, std::string doc, DALIDataType default_value);

  /**  Adds an optional argument of type DALIDataType without a default value
   *
   * If the arg name starts with an underscore, it will be marked hidden, which
   * makes it not listed in the docs.
   */
  OpSchema &AddOptionalTypeArg(std::string_view name, std::string doc);

  inline OpSchema &AddOptionalArg(std::string_view name, std::string doc,
                                      const char *default_value) {
    return AddOptionalArg(name, std::move(doc), std::string(default_value), false);
  }

  inline OpSchema &AddOptionalArg(std::string_view name, std::string doc,
                                  std::string_view default_value) {
    return AddOptionalArg(name, std::move(doc), std::string(default_value), false);
  }

  /**  Adds an optional vector argument to op
   *
   * If the arg name starts with an underscore, it will be marked hidden, which
   * makes it not listed in the docs.
   */
  template <typename T>
  inline OpSchema &AddOptionalArg(std::string_view name, std::string doc,
                                  std::vector<T> default_value,
                                  bool enable_tensor_input = false,
                                  bool support_per_frame_input = false) {
    using S = argument_storage_t<T>;
    auto value = Value::construct(detail::convert_vector<S>(default_value));
    auto &arg = AddArgumentImpl(name,
                                type2id<std::vector<T>>::value,
                                std::move(value),
                                std::move(doc));
    arg.tensor = enable_tensor_input;
    arg.per_frame = support_per_frame_input;
    return *this;
  }

  /** Adds a random seed argument to the operator.
   *
   * If not provided, the seed will be selected automatically.
   * Adding a random seed implies statefulness of the operator.
   */
  OpSchema &AddRandomSeedArg();

  /** Adds a random state argument to the operator.
   *
   * This argument is used to pass the initial state of the random number generator
   * to random operators in Dynamic Mode. It accepts a 1D tensor of uint32_t values.
   * It is not advertised in the Python API and is automatically hidden from 
   * documentation (by using a leading underscore).
   */
  OpSchema &AddRandomStateArg();

  /**  Marks an argument as deprecated in favor of a new argument
   *
   * Providing renamed_to means the argument has been renamed and we can safely
   * propagate the value to the new argument name.
   */
  OpSchema &DeprecateArgInFavorOf(std::string_view arg_name, std::string renamed_to,
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
  OpSchema &DeprecateArg(std::string_view arg_name, bool removed = true,
                         std::string msg = {});

  /**
   * @brief Sets a function that infers whether the op can
   *        be executed in-place depending on the ops specification.
   */
  OpSchema &InPlaceFn(SpecFunc f);

  /** Sets a parent (which could be used as a storage of default parameters)
   *
   * Does not support cyclic dependency. There can be multiple parents and the lookup is transitive.
   * Only arguments are inherited, inputs and outputs are not.
   */
  OpSchema &AddParent(std::string parent_name);

  /**  Notes that this operator should not be pruned from a graph even if its outputs are unused. */
  OpSchema &NoPrune();

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
  OpSchema &PassThrough(const std::map<int, int> &inout);

  /**
   * @brief Informs that the operator passes through data unchanged, sharing the allocation
   *        from input to output.
   *
   * The data is passed on sample basis, allowing to mix any input to any output.
   */
  OpSchema &SamplewisePassThrough();

  /** Get parent schemas (non-recursive) */
  const vector<std::string> &GetParentNames() const;

  const vector<const OpSchema *> &GetParents() const;

  /** Get the docstring of the operator - provided by DocStr in the schema definition. */
  string Dox() const;

  /** Return true wether the default input docs can be used */
  bool CanUseAutoInputDox() const;

  /**
   * @brief Whether the docstring for kwargs should be automatically generated and appended to the
   *        one provided in CallDocStr.
   */
  bool AppendKwargsSection() const;

  /**  Return true when `__call__` docstring was explicitly set
   *
   * Should be considered as highest preference
   */
  bool HasCallDox() const;

  /** Get the documentation for Operator __call__ signature provided by CallDocStr. */
  std::string GetCallDox() const;

  /** Check if this operator has input docstrings provided */
  bool HasInputDox() const;

  /**
   * @brief List all the inputs that should appear in `__call__` signature based on the input
   *        docs that were specified. Requires HasInputDox() to return true
   *
   */
  std::string GetCallSignatureInputs() const;

  /** Get the docstring name of the input at given index. */
  std::string GetInputName(int input_idx) const;

  /** Get the docstring type of the input at given index. */
  std::string GetInputType(int input_idx) const;

  /** Get the docstring text of the input at given index. */
  std::string GetInputDox(int input_idx) const;

  /** Get the maximal number of accepted inputs. */
  int MaxNumInput() const;

  /** Get the minimal number of required inputs. */
  int MinNumInput() const;

  /** Get the number of static outputs, see also CalculateOutputs and CalculateAdditionalOutputs */
  int NumOutput() const;

  /** Whether this operator accepts ONLY sequences as inputs */
  bool IsSequenceOperator() const;

  /** Whether this operator accepts sequences as inputs */
  bool AllowsSequences() const;

  /** Whether this operator accepts volumes as inputs */
  bool SupportsVolumetric() const;

  /**  Whether this operator is internal to DALI backend (and shouldn't be exposed in Python API) */
  bool IsInternal() const;

  /** Whether this operator is stateful.
   *
   * Returns the statefulness of the operator. If it wasn't explicitly set, then the operator is
   * considered stateful if any of its parents is stateful.
   */
  bool IsStateful() const;

  /** Whether this operator doc should not be visible (but the Op is exposed in Python API) */
  bool IsDocHidden() const;

  /**
   * Whether this operator doc should be visible without documenting any parameters.
   * Useful for deprecated ops.
   */
  bool IsDocPartiallyHidden() const;

  /** Whether this operator is deprecated. */
  bool IsDeprecated() const;

  /** What operator replaced the current one. */
  const std::string &DeprecatedInFavorOf() const;

  /** Additional deprecation message */
  const std::string &DeprecationMessage() const;

  /** Whether given argument is deprecated. */
  bool IsDeprecatedArg(std::string_view arg_name) const;

  /** Information about the argument deprecation - error message, renaming, removal, etc. */
  const ArgumentDeprecation &DeprecatedArgInfo(std::string_view arg_name) const;

  /** Check whether this operator calculates number of outputs statically
   *
   * @return false if static, true if dynamic
   */
  bool HasOutputFn() const;

  /** Check whether this operator won't be pruned out of graph even if not used. */
  bool IsNoPrune() const;

  bool IsSerializable() const;

  /**  Returns the index of the output to which the input is passed.
   *
   * @param strict consider only fully passed through batches
   * @return Output indicies or empty vector if given input is not passed through.
   */
  std::vector<int> GetPassThroughOutputIdx(int input_idx, const OpSpec &spec,
                                                      bool strict = true) const;

  /** Is the input_idx passed through to output_idx */
  bool IsPassThrough(int input_idx, int output_idx, bool strict = true) const;

  /** Does this operator pass through any data? */
  bool HasPassThrough() const;

  /** Does this operator pass through any data as a whole batch to batch? */
  bool HasStrictPassThrough() const;

  /** Does this operator pass through any data by the means of sharing individual samples? */
  bool HasSamplewisePassThrough() const;

  /** Return the static number of outputs or calculate regular outputs using output_fn */
  int CalculateOutputs(const OpSpec &spec) const;

  /** Calculate the number of additional outputs obtained from additional_outputs_fn */
  int CalculateAdditionalOutputs(const OpSpec &spec) const;

  bool SupportsInPlace(const OpSpec &spec) const;

  void CheckArgs(const OpSpec &spec) const;

  /** Get default value of optional or internal argument. The default value must be declared */
  template <typename T>
  inline T GetDefaultValueForArgument(std::string_view s) const;

  /** Checks if the argument with the given name is defined and not required */
  bool HasOptionalArgument(std::string_view name) const;

  /** Checks if the argument with the given name is defined and marked as internal */
  bool HasInternalArgument(std::string_view name) const;

  /** Finds default value for a given argument
   *
   * @return A pair of the defining schema and the value
   */
  const Value *FindDefaultValue(std::string_view arg_name) const;

  /** Checks whether the schema defines an argument with the given name
   *
   * @param include_internal - returns `true` also for internal/implicit arugments
   * @param local_only       - doesn't look in parent schemas
   */
  bool HasArgument(std::string_view name, bool include_internal = false) const;

  /** Returns true if the operator has a "_random_state" tensor argument. */
  bool HasRandomStateArg() const;

  /** Returns true if the operator has a "seed" argument. */
  bool HasRandomSeedArg() const;

  /** Get docstring for operator argument of given name (Python Operator Kwargs). */
  const std::string &GetArgumentDox(std::string_view name) const;

  /** Get enum representing type of argument of given name. */
  DALIDataType GetArgumentType(std::string_view name) const;

  const ArgumentDef &GetArgument(std::string_view name) const;

  /** Check if the argument has a default value.
   *
   * Required arguments always return false.
   * Internal arguments always return true.
   */
  bool HasArgumentDefaultValue(std::string_view name) const;

  /**
   * @brief Get default value of optional argument represented as python-compatible repr string.
   *        Not allowed for internal arguments.
   */
  std::string GetArgumentDefaultValueString(std::string_view name) const;

  /** Get names of all required, optional, and deprecated arguments.
   *
   * @param include_hidden - if true, includes hidden arguments
   */
  std::vector<std::string> GetArgumentNames(bool include_hidden = false) const;
  bool IsTensorArgument(std::string_view name) const;
  bool ArgSupportsPerFrameInput(std::string_view arg_name) const;

  bool IsDefault() const { return default_; }

 private:
  struct DefaultSchemaTag {};
  /** Populates the default schema with common arguments */
  explicit OpSchema(DefaultSchemaTag);

  static inline bool ShouldHideArgument(std::string_view name) {
    return name.size() && name[0] == '_';
  }

  const ArgumentDef *FindTensorArgument(std::string_view name) const;

  template <typename Pred>
  const ArgumentDef *FindArgument(std::string_view name, Pred &&pred) const {
    if (auto *arg = FindArgument(name))
      return pred(*arg) ? arg : nullptr;
    else
      return nullptr;
  }

  const ArgumentDef *FindArgument(std::string_view name) const {
    auto &args = GetFlattenedArguments();
    auto it = args.find(name);
    if (it == args.end())
      return nullptr;
    return it->second;
  }

  void CheckArgument(std::string_view s);

  void CheckInputIndex(int index) const;

  std::string DefaultDeprecatedArgMsg(std::string_view arg_name, std::string_view renamed_to,
                                      bool removed) const;

  /** Add internal argument to schema. It always has a value. */
  template <typename T>
  void AddInternalArg(std::string_view name, std::string doc, T value) {
    auto &arg = AddArgumentImpl(name, type2id<T>::value, Value::construct(value), std::move(doc));
    arg.hidden = true;
    arg.internal = true;
  }

  ArgumentDef &AddArgumentImpl(std::string_view name);

  ArgumentDef &AddArgumentImpl(std::string_view name,
                               DALIDataType type,
                               std::unique_ptr<Value> default_value,
                               std::string doc);

  /** Initialize the module_path_ and operator_name_ fields based on the schema name. */
  void InitNames();

  std::string dox_;
  /// The name of the schema
  std::string name_;
  /// The module path for the operator
  std::vector<std::string> module_path_;
  /// The PascalCase name of the operator (without the module path)
  std::string operator_name_;

  ////////////////////////////////////////////////////////////////////////////
  // Inputs, outputs and arguments

  /// All locally defined arguments
  std::map<std::string, ArgumentDef, std::less<>> arguments_;

  mutable
  detail::LazyValue<std::map<std::string, const ArgumentDef *, std::less<>>> flattened_arguments_;
  mutable int circular_inheritance_detector_ = 0;

  std::map<std::string, const ArgumentDef *, std::less<>> &GetFlattenedArguments() const;

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
  mutable detail::LazyValue<std::vector<const OpSchema *>> parents_;

  ////////////////////////////////////////////////////////////////////////////
  // Documentation-related
  bool support_volumetric_ = false;
  bool allow_sequences_ = false;
  bool is_sequence_operator_ = false;

  bool is_internal_ = false;
  bool is_doc_hidden_ = false;
  bool is_doc_partially_hidden_ = false;
  mutable std::optional<bool> is_stateful_;

  /// Custom docstring, if not empty should be used in place of input_dox_ descriptions
  std::string call_dox_ = {};

  /// Whether to append kwargs section to __call__ docstring. Off by default,
  /// can be turned on for call_dox_ specified manually
  bool append_kwargs_section_ = false;

  ////////////////////////////////////////////////////////////////////////////
  // Internal flags
  bool no_prune_ = false;
  bool serializable_ = true;
  const bool default_ = false;

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
inline T OpSchema::GetDefaultValueForArgument(std::string_view name) const {
  const Value *v = FindDefaultValue(name);
  if (!v) {
    (void)GetArgument(name);  // throw an error if the argument is undefined

    // otherwise throw a different error
    throw std::invalid_argument(make_string(
      "The argument \"", name, "\" in operator \"", this->name(),
      "\" doesn't have a default value."));
  }

  using S = argument_storage_t<T>;
  const ValueInst<S> *vS = dynamic_cast<const ValueInst<S> *>(v);
  if (!vS) {
    throw std::invalid_argument(make_string(
      "Unexpected type of the default value for argument \"", name, "\" of schema \"",
      this->name(), "\""));
  }
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
