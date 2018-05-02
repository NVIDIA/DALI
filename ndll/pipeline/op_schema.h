// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OP_SCHEMA_H_
#define NDLL_PIPELINE_OP_SCHEMA_H_

#include <functional>
#include <map>
#include <string>
#include <set>
#include <vector>
#include <utility>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/argument.h"

namespace ndll {

class OpSpec;

class OpSchema {
 public:
  typedef std::function<int(const OpSpec &spec)> SpecFunc;

  inline OpSchema()
    : allow_multiple_input_sets_(false) {
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

  inline ~OpSchema()          { delete [] getParentName(); }

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
  inline void setParentName(const char *parentName) {
    delete [] getParentName();
    parentName_ = !parentName || !parentName[0]? NULL :
                  strcpy(new char [strlen(parentName) + 1], parentName);
  }

  inline const char *getParentName() const          { return parentName_; }

  inline string Dox() const {
    std::string ret = dox_;
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

  inline void CheckArgs(std::vector<std::string> vec) const {
    std::set<std::string> req_arguments_left;
    for (auto& arg_pair : arguments_) {
      req_arguments_left.insert(arg_pair.first);
    }
    for (std::string s : vec) {
      NDLL_ENFORCE(arguments_.find(s) != arguments_.end() ||
          OptionalArgumentExists(s) ||
          s == "device",
          "Got an unexpected argument \"" + s + "\"");
      std::set<std::string>::iterator it = req_arguments_left.find(s);
      if (it != req_arguments_left.end()) {
        req_arguments_left.erase(it);
      }
    }
    if (!req_arguments_left.empty()) {
      std::string ret = "Not all required arguments were specified. "
        "Please specify values for arguments: ";
      for (auto& str : req_arguments_left) {
        ret += "\"" + str + "\", ";
      }
      ret.erase(ret.size()-2);
      ret += ".";
      NDLL_FAIL(ret);
    }
  }

  template<typename T>
  inline T GetDefaultValueForOptionalArgument(const std::string &s) const {
    const bool argFound = OptionalArgumentExists(s);
    NDLL_ENFORCE(argFound ||
        internal_arguments_.find(s) != internal_arguments_.end(),
        "Default value does not exist for argument \"" + s + "\"");
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
  }

  bool OptionalArgumentExists(const std::string &s) const {
    return optional_arguments_.find(s) != optional_arguments_.end();
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
  SpecFunc output_fn_, in_place_fn_, additional_outputs_fn_;

  int min_num_input_ = 0, max_num_input_ = 0;
  int num_output_ = 0;

  bool allow_multiple_input_sets_;
  char *parentName_ = NULL;

  std::map<std::string, std::string> arguments_;
  std::map<std::string, std::pair<std::string, Value*> > optional_arguments_;
  std::map<std::string, std::pair<std::string, Value*> > internal_arguments_;
};

class SchemaRegistry {
 public:
  static OpSchema& RegisterSchema(const std::string &name, const char *parentName = NULL) {
    auto &schema_map = registry();
    NDLL_ENFORCE(schema_map.count(name) == 0, "OpSchema already "
        "registered for operator '" + name + "'. NDLL_OPERATOR_SCHEMA(op) "
        "should only be called once per op.");

    // Insert the op schema and return a reference to it
    OpSchema &schema = schema_map[name];
    schema.setParentName(parentName);
    return schema;
  }

  static const OpSchema& GetSchema(const std::string &name) {
    auto &schema_map = registry();
    auto it = schema_map.find(name);
    NDLL_ENFORCE(it != schema_map.end(), "Schema for op '" +
        name + "' not registered");
    return it->second;
  }

  static const OpSchema *GetSchema(const char *pName) {
    if (!pName)
      return NULL;

    auto &schema_map = registry();
    auto it = schema_map.find(pName);
    return it != schema_map.end()?  &it->second : NULL;
  }

 private:
  inline SchemaRegistry() {}

  static std::map<string, OpSchema>& registry();
};

#define NDLL_OPERATOR_SCHEMA_REG(OpName, ParentOpName)      \
  int NDLL_OPERATOR_SCHEMA_REQUIRED_FOR_##OpName() {        \
    return 42;                                              \
  }                                                         \
  static OpSchema* ANONYMIZE_VARIABLE(OpName) =             \
    &SchemaRegistry::RegisterSchema(#OpName, ParentOpName)

#define NDLL_OPERATOR_SCHEMA(OpName)                            \
      NDLL_OPERATOR_SCHEMA_REG(OpName, NULL)

#define NDLL_OPERATOR_SCHEMA_WITH_PARENT(OpName, ParentOpName)  \
      NDLL_OPERATOR_SCHEMA_REG(OpName, #ParentOpName)

}  // namespace ndll

#endif  // NDLL_PIPELINE_OP_SCHEMA_H_
