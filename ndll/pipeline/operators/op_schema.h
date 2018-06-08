// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_OP_SCHEMA_H_
#define NDLL_PIPELINE_OPERATORS_OP_SCHEMA_H_

#include <functional>
#include <map>
#include <string>
#include <set>
#include <vector>
#include <utility>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/operators/argument.h"

namespace ndll {

class OpSpec;

class OpSchema {
 public:
  typedef std::function<int(const OpSpec &spec)> SpecFunc;

  explicit inline OpSchema(const std::string &name)
    : name_(name),
      allow_multiple_input_sets_(false) {
    // Fill internal arguments
    internal_arguments_["num_threads"] = std::make_pair("Number of CPU threads in a thread pool",
        Value::construct(-1));
    internal_arguments_["batch_size"] = std::make_pair("Batch size",
        Value::construct(-1));
    internal_arguments_["num_input_sets"] = std::make_pair("Number of input sets given to an Op",
        Value::construct(1));
    internal_arguments_["device"] = std::make_pair("Device on which the Op is run",
        Value::construct(std::string("cpu")));
    internal_arguments_["inplace"] = std::make_pair("Whether Op can be run in place",
        Value::construct(false));
    internal_arguments_["seed"] = std::make_pair("Random seed",
        Value::construct(1234));
  }

  inline ~OpSchema() = default;

  /**
   * @brief Returns the name of this operator.
   */
  inline const std::string& name() const {
    return name_;
  }

  /**
   * @brief Sets the doc string for this operator.
   */
  inline OpSchema& DocStr(const string &dox) {
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
  inline OpSchema& OutputFn(SpecFunc f) {
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
  inline OpSchema& AdditionalOutputsFn(SpecFunc f) {
    additional_outputs_fn_ = f;
    return *this;
  }

  /**
   * @brief Sets the number of inputs that the op can receive.
   */
  inline OpSchema& NumInput(int n) {
    NDLL_ENFORCE(n >= 0);
    max_num_input_ = n;
    min_num_input_ = n;
    return *this;
  }

  /**
   * @brief Sets the min and max number of inputs the op can receive.
   */
  inline OpSchema& NumInput(int min, int max) {
    NDLL_ENFORCE(min <= max);
    NDLL_ENFORCE(min >= 0);
    NDLL_ENFORCE(max >= 0);
    min_num_input_ = min;
    max_num_input_ = max;
    return *this;
  }

  /**
   * @brief Sets the number of outputs that the op can receive.
   */
  inline OpSchema& NumOutput(int n) {
    NDLL_ENFORCE(n >= 0);
    num_output_ = n;
    return *this;
  }

  /**
   * @brief Notes that multiple input sets can be used with this op
   */
  inline OpSchema& AllowMultipleInputSets() {
    allow_multiple_input_sets_ = true;
    return *this;
  }

  /**
   * @brief Adds a required argument to op
   */
  inline OpSchema& AddArg(const std::string &s, const std::string &doc) {
    CheckArgument(s);
    arguments_[s] = doc;
    return *this;
  }

  /**
   * @brief Adds an optional non-vector argument to op
   */
  template <typename T>
  inline typename std::enable_if<
    !is_vector<T>::value && !is_array<T>::value,
    OpSchema&>::type
  AddOptionalArg(const std::string &s, const std::string &doc, T default_value) {
    CheckArgument(s);
    std::string stored_doc = doc + " (default value: " + to_string(default_value) + ")";
    Value * to_store = Value::construct(default_value);
    optional_arguments_[s] = std::make_pair(stored_doc, to_store);
    return *this;
  }

  /**
   * @brief Adds an optional vector argument to op
   */
  template <typename T>
  inline OpSchema& AddOptionalArg(const std::string &s, const std::string &doc,
                                  std::vector<T> default_value) {
    CheckArgument(s);
    std::string stored_doc = doc + " (default value: " + to_string(default_value) + ")";
    Value * to_store = Value::construct(std::vector<T>(default_value));
    optional_arguments_[s] = std::make_pair(stored_doc, to_store);
    return *this;
  }

  /**
   * @brief Sets a function that infers whether the op can
   * be executed in-place depending on the ops specification.
   */
  inline OpSchema& InPlaceFn(SpecFunc f) {
    REPORT_FATAL_PROBLEM("In-place op support not yet implemented.");
    return *this;
  }

  /**
   * @brief Sets a parent (which could be used as a storage of default parameters
   */
  inline OpSchema& AddParent(const std::string &parentName) {
    parents_.push_back(parentName);
    return *this;
  }

  inline OpSchema& SetName(const std::string &name) {
    name_ = name;
    return *this;
  }

  inline string Name() const {
    return name_;
  }

  inline const vector<std::string>& GetParents() const {
    return parents_;
  }

  inline bool HasParent() const {
    return parents_.size() > 0;
  }

  inline string Dox() const {
    std::string ret = Name();
    ret += "\n\nOverview\n----------\n";
    ret += dox_;
    ret += "\n\nParameters\n----------\n";
    for (auto arg_pair : arguments_) {
      ret += arg_pair.first + " : " + arg_pair.second + "\n";
    }
    ret += "\n\nOptional Parameters\n-------------------\n";
    for (auto arg_pair : optional_arguments_) {
      ret += arg_pair.first + " : " + arg_pair.second.first + "\n";
    }
    return ret;
  }

  inline int MaxNumInput() const {
    return max_num_input_;
  }

  inline int MinNumInput() const {
    return min_num_input_;
  }

  inline int NumOutput() const {
    return num_output_;
  }

  inline bool AllowsMultipleInputSets() const {
    return allow_multiple_input_sets_;
  }

  inline bool HasOutputFn() const {
    return static_cast<bool>(output_fn_);
  }

  int CalculateOutputs(const OpSpec &spec) const;

  int CalculateAdditionalOutputs(const OpSpec &spec) const {
    if (!additional_outputs_fn_) return 0;
    return additional_outputs_fn_(spec);
  }

  inline bool SupportsInPlace(const OpSpec &spec) const {
    if (!in_place_fn_) return false;
    return in_place_fn_(spec);
  }

  void CheckArgs(const OpSpec &spec) const;

  inline const OpSchema& GetSchemaWithArgument(const string& name) const;

  inline bool OptionalArgumentExists(const std::string &s, const bool local_only = false) const;

  template<typename T>
  inline T GetDefaultValueForOptionalArgument(const std::string &s) const;

  inline bool HasRequiredArgument(const std::string &name) const {
    return arguments_.find(name) != arguments_.end();
  }

  inline bool HasOptionalArgument(const std::string &name) const {
    return optional_arguments_.find(name) != optional_arguments_.end();
  }

  inline bool HasArgument(const string &name) const {
    return HasRequiredArgument(name) || HasOptionalArgument(name);
  }

 private:
  inline bool CheckArgument(const std::string &s) {
    NDLL_ENFORCE(arguments_.find(s) == arguments_.end(),
                 "Argument \"" + s + "\" already added to the schema");
    NDLL_ENFORCE(!OptionalArgumentExists(s),
                 "Argument \"" + s + "\" already added to the schema");
    NDLL_ENFORCE(internal_arguments_.find(s) == internal_arguments_.end(),
                 "Argument name \"" + s + "\" is reserved for internal use");
    return true;
  }

  string dox_;
  string name_;
  SpecFunc output_fn_, in_place_fn_, additional_outputs_fn_;

  int min_num_input_ = 0, max_num_input_ = 0;
  int num_output_ = 0;

  bool allow_multiple_input_sets_;
  vector<string> parents_;

  std::map<std::string, std::string> arguments_;
  std::map<std::string, std::pair<std::string, Value*> > optional_arguments_;
  std::map<std::string, std::pair<std::string, Value*> > internal_arguments_;
};

class SchemaRegistry {
 public:
  static OpSchema& RegisterSchema(const std::string &name) {
    auto &schema_map = registry();
    NDLL_ENFORCE(schema_map.count(name) == 0, "OpSchema already "
        "registered for operator '" + name + "'. NDLL_SCHEMA(op) "
        "should only be called once per op.");

    // Insert the op schema and return a reference to it
    schema_map.insert(std::make_pair(name, OpSchema(name)));
    return schema_map.at(name);
  }

  static const OpSchema& GetSchema(const std::string &name) {
    auto &schema_map = registry();
    auto it = schema_map.find(name);
    NDLL_ENFORCE(it != schema_map.end(), "Schema for op '" +
        name + "' not registered");
    return it->second;
  }

 private:
  inline SchemaRegistry() {}

  static std::map<string, OpSchema>& registry();
};

inline string GetSchemaWithArg(const string& start, const string& arg) {
  const OpSchema& s = SchemaRegistry::GetSchema(start);

  // Found locally, return immediately
  if (s.OptionalArgumentExists(arg, true)) {
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
  const bool argFound = OptionalArgumentExists(s, true);

  if (argFound || internal_arguments_.find(s) != internal_arguments_.end()) {
    Value * v;
    if (argFound) {
      auto arg_pair = *optional_arguments_.find(s);
      v = arg_pair.second.second;
    } else {
      auto arg_pair = *internal_arguments_.find(s);
      v = arg_pair.second.second;
    }
    ValueInst<T> * vT = dynamic_cast<ValueInst<T>*>(v);
    NDLL_ENFORCE(vT != nullptr, "Unexpected type of the default value for argument \"" + s + "\"");
    return vT->Get();
  } else {
    // get the parent schema that has the optional argument and return from there
    string tmp = GetSchemaWithArg(Name(), s);
    const OpSchema& schema = SchemaRegistry::GetSchema(tmp);
    return schema.template GetDefaultValueForOptionalArgument<T>(s);
  }
}

bool OpSchema::OptionalArgumentExists(const std::string &s,
                                      const bool local_only) const {
  // check just this schema for the argument
  if (local_only) {
    return optional_arguments_.find(s) != optional_arguments_.end();
  } else {
    // recurse through this schema and all parents (through inheritance tree)
    string tmp = GetSchemaWithArg(Name(), s);
    if (tmp.empty()) return false;

    const OpSchema &schema = SchemaRegistry::GetSchema(tmp);
    return schema.OptionalArgumentExists(s, true);
  }
}

#define NDLL_SCHEMA_REG(OpName)      \
  int NDLL_OPERATOR_SCHEMA_REQUIRED_FOR_##OpName() {        \
    return 42;                                              \
  }                                                         \
  static OpSchema* ANONYMIZE_VARIABLE(OpName) =             \
    &SchemaRegistry::RegisterSchema(#OpName)

#define NDLL_SCHEMA(OpName)                            \
      NDLL_SCHEMA_REG(OpName)

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_OP_SCHEMA_H_
