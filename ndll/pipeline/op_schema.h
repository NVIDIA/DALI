// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OP_SCHEMA_H_
#define NDLL_PIPELINE_OP_SCHEMA_H_

#include <functional>
#include <map>
#include <string>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/op_spec.h"

namespace ndll {

class OpSchema {
 public:
  typedef std::function<int(const OpSpec &spec)> SpecFunc;

  inline OpSchema() {}
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
   * @brief Sets a function that infers whether the op can
   * be executed in-place depending on the ops specification.
   */
  inline OpSchema& InPlaceFn(SpecFunc f) {
    NDLL_FAIL("In-place op support not yet implemented.");
    return *this;
  }

  inline string Dox() const {
    return dox_;
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

 private:
  string dox_;
  SpecFunc output_fn_, in_place_fn_;

  int min_num_input_ = 0, max_num_input_ = 0;
  int min_num_output_ = 0, max_num_output_ = 0;
};

class SchemaRegistry {
 public:
  static OpSchema& RegisterSchema(std::string name) {
    auto &schema_map = registry();
    NDLL_ENFORCE(schema_map.count(name) == 0, "OpSchema already "
        "registered for operator '" + name + "'. OPERATOR_SCHEMA(op) "
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

#define OPERATOR_SCHEMA(OpName)                       \
  int OPERATOR_SCHEMA_REQUIRED_FOR_##OpName() {       \
    return 42;                                        \
  }                                                   \
  static OpSchema* ANONYMIZE_VARIABLE(OpName) =       \
    &SchemaRegistry::RegisterSchema(#OpName)          \

}  // namespace ndll

#endif  // NDLL_PIPELINE_OP_SCHEMA_H_
