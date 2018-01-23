// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_READER_PARSER_TF_FEATURE_H_
#define NDLL_PIPELINE_OPERATORS_READER_PARSER_TF_FEATURE_H_

#ifdef NDLL_BUILD_PROTO3

#include <vector>
#include <string>

#include "ndll/common.h"
#include "ndll/pipeline/argument.h"
#include "ndll/pipeline/op_spec.h"
#include "ndll/pipeline/operators/reader/parser/parser.h"
#include "ndll/pipeline/operators/reader/parser/example.pb.h"

namespace ndll {

namespace TFUtil {

enum FeatureType {
  int64,
  string,
  float32 };

class Feature {
 public:
  struct Value {
    std::string str;
    int64_t int64;
    float float32;
  };

  Feature(std::vector<Index> shape, FeatureType type, Value val) {
    has_shape_ = true;
    shape_ = shape;
    type_ = type;
    val_ = val;
  }

  Feature(FeatureType type, Value val) {
    has_shape_ = false;
    type_ = type;
    val_ = val;
  }

  FeatureType GetType() { return type_; }
  bool HasShape() { return has_shape_; }
  const std::vector<Index>& Shape() const {
    return shape_;
  }
  const Value GetValue() { return val_; }

  template<typename T>
  T GetDefaultValue() {
    if (std::is_same<T, std::string>::value()) {
      NDLL_ENFORCE(GetType() == FeatureType::string,
          "Requested invalid type from the Feature");
      return val_.str;
    } else if (std::is_same<T, int64_t>::value()) {
      NDLL_ENFORCE(GetType() == FeatureType::int64,
          "Requested invalid type from the Feature");
      return val_.int64;
    } else if (std::is_same<T, float>::value()) {
      NDLL_ENFORCE(GetType() == FeatureType::float32,
          "Requested invalid type from the Feature");
      return val_.float32;
    } else {
      NDLL_FAIL("Requested invalid type from the feature");
      return T();
    }
  }

  std::string ToString() const {
    std::string ret = "";
    if (has_shape_) {
      ret += "FixedLenFeature {";
      ret += to_string(shape_);
      ret += ",";
    } else {
      ret += "VarLenFeature {";
    }
    ret += to_string(type_);
    ret += ",";
    switch (type_) {
      case FeatureType::string:
        ret += to_string(val_.str);
        break;
      case FeatureType::int64:
        ret += to_string(val_.int64);
        break;
      case FeatureType::float32:
        ret += to_string(val_.float32);
        break;
    }
    ret += " }";
    return ret;
  }

 private:
  bool has_shape_;
  std::vector<Index> shape_;
  FeatureType type_;
  Value val_;
};

}  // namespace TFUtil

}  // namespace ndll

#endif  // NDLL_BUILD_PROTO3

#endif  // NDLL_PIPELINE_OPERATORS_READER_PARSER_TF_FEATURE_H_
