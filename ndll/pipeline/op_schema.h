// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OP_SCHEMA_H_
#define NDLL_PIPELINE_OP_SCHEMA_H_

#include <functional>
#include <map>
#include <string>
#include <set>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/op_spec.h"

namespace ndll {

class OpSchema {
 public:
  typedef std::function<int(const OpSpec &spec)> SpecFunc;

  inline OpSchema()
    : allow_multiple_input_sets_(false) {}
  inline ~OpSchema() = default;

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
    max_num_output_ = n;
    min_num_output_ = n;
    return *this;
  }

  /**
   * @brief Sets the min and max number of outputs the op can receive.
   */
  inline OpSchema& NumOutput(int min, int max) {
    NDLL_ENFORCE(min <= max);
    NDLL_ENFORCE(min >= 0);
    NDLL_ENFORCE(max >= 0);
    min_num_output_ = min;
    max_num_output_ = max;
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
  inline OpSchema& AddArg(std::string s, std::string doc) {
    NDLL_ENFORCE(arguments_.find(s) == arguments_.end(), "Argument \"" + s +
        "\" already added to the schema");
    NDLL_ENFORCE(optional_arguments_.find(s) == optional_arguments_.end(), "Argument \"" + s +
        "\" already added to the schema");
    arguments_[s] = doc;
    return *this;
  }

  /**
   * @brief Adds an optional argument to op
   */
  inline OpSchema& AddOptionalArg(std::string s, std::string doc) {
    NDLL_ENFORCE(arguments_.find(s) == arguments_.end(), "Argument \"" + s +
        "\" already added to the schema");
    NDLL_ENFORCE(optional_arguments_.find(s) == optional_arguments_.end(), "Argument \"" + s +
        "\" already added to the schema");
    optional_arguments_[s] = doc;
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

  inline string Dox() const {
    std::string ret = dox_;
    ret += "\n\nParameters\n----------\n";
    for (auto arg_pair : arguments_) {
      ret += arg_pair.first + " : " + arg_pair.second + "\n";
    }
    ret += "\n\nOptional Parameters\n-------------------\n";
    for (auto arg_pair : optional_arguments_) {
      ret += arg_pair.first + " : " + arg_pair.second + "\n";
    }
    return ret;
  }

  inline int MaxNumInput() const {
    return max_num_input_;
  }

  inline int MinNumInput() const {
    return min_num_input_;
  }

  inline int MaxNumOutput() const {
    return max_num_output_;
  }

  inline int MinNumOutput() const {
    return min_num_output_;
  }

  inline bool AllowsMultipleInputSets() const {
    return allow_multiple_input_sets_;
  }

  inline bool HasOutputFn() const {
    return static_cast<bool>(output_fn_);
  }

  inline int CalculateOutputs(const OpSpec &spec) const {
    if (!output_fn_) {
      return max_num_output_;
    } else {
      return output_fn_(spec);
    }
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
          optional_arguments_.find(s) != optional_arguments_.end() ||
          s == "device",
          "Got an unexpected argument \"" + s + "\"");
      std::set<std::string>::iterator it = req_arguments_left.find(s);
      if (it != req_arguments_left.end()) {
        req_arguments_left.erase(it);
      }
    }
    if (!req_arguments_left.empty()) {
      std::string ret = "Not all required arguments were specified. Please specify values for arguments: ";
      for (auto& str : req_arguments_left) {
        ret += "\"" + str + "\", ";
      }
      ret.erase(ret.size()-2);
      ret += ".";
      NDLL_FAIL(ret);
    }
  }

 private:
  string dox_;
  SpecFunc output_fn_, in_place_fn_;

  int min_num_input_ = 0, max_num_input_ = 0;
  int min_num_output_ = 0, max_num_output_ = 0;

  bool allow_multiple_input_sets_;

  std::map<std::string, std::string> arguments_;
  std::map<std::string, std::string> optional_arguments_;
};

class SchemaRegistry {
 public:
  static OpSchema& RegisterSchema(std::string name) {
    auto &schema_map = registry();
    NDLL_ENFORCE(schema_map.count(name) == 0, "OpSchema already "
        "registered for operator '" + name + "'. NDLL_OPERATOR_SCHEMA(op) "
        "should only be called once per op.");

    // Insert the op schema and return a reference to it
    return schema_map[name];
  }

  static OpSchema& GetSchema(std::string name) {
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

#define NDLL_OPERATOR_SCHEMA(OpName)                       \
  int NDLL_OPERATOR_SCHEMA_REQUIRED_FOR_##OpName() {       \
    return 42;                                        \
  }                                                   \
  static OpSchema* ANONYMIZE_VARIABLE(OpName) =       \
    &SchemaRegistry::RegisterSchema(#OpName)          \

}  // namespace ndll

#endif  // NDLL_PIPELINE_OP_SCHEMA_H_
