// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_READER_PARSER_TF_FEATURE_H_
#define NDLL_PIPELINE_OPERATORS_READER_PARSER_TF_FEATURE_H_

#ifdef NDLL_BUILD_PROTO3

#include <vector>
#include <string>

#include "ndll/common.h"
#include "ndll/pipeline/operators/argument.h"
#include "ndll/pipeline/operators/op_spec.h"
#include "ndll/pipeline/operators/reader/parser/parser.h"
#include "ndll/pipeline/proto/ndll_proto_utils.h"

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

  Feature() {}

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

  FeatureType GetType() const { return type_; }
  bool HasShape() const { return has_shape_; }
  const std::vector<Index>& Shape() const {
    return shape_;
  }
  const Value GetValue() const { return val_; }

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

  std::string SerializeType() const {
    return "TFRecord";
  }

  ndll_proto::Argument * SerializeToProtobuf(ndll_proto::Argument *arg) const {
    arg->set_type(SerializeType());
    arg->set_is_vector(false);

    // set the datatype of the record
    auto *type_arg = arg->add_extra_args();
    type_arg->set_name("type");
    ndll::SerializeToProtobuf(static_cast<int>(GetType()), type_arg);

    // has_shape
    auto *has_shape_arg = arg->add_extra_args();
    has_shape_arg->set_name("has_shape");
    ndll::SerializeToProtobuf(HasShape(), has_shape_arg);

    // set the shape
    auto *shape_arg = arg->add_extra_args();
    shape_arg->set_name("shape");
    shape_arg->set_is_vector(false);
    auto& shape = Shape();
    for (size_t i = 0; i < shape.size(); ++i) {
      ndll::SerializeToProtobuf(shape[i], shape_arg);
    }

    // set the default value
    auto *default_arg = arg->add_extra_args();
    default_arg->set_name("default_value");
    switch (GetType()) {
      case TFUtil::FeatureType::int64:
        ndll::SerializeToProtobuf(GetValue().int64, default_arg);
        break;
      case TFUtil::FeatureType::string:
        ndll::SerializeToProtobuf(GetValue().str, default_arg);
        break;
      case TFUtil::FeatureType::float32:
        ndll::SerializeToProtobuf(GetValue().float32, default_arg);
        break;
      default:
        NDLL_FAIL("Unknown TFUtil::FeatureType value");
    }
    return arg;
  }

  static Feature DeserializeFromProtobuf(const ndll_proto::Argument& arg) {
    // type argument
    ndll_proto::Argument type_arg = arg.extra_args(0);
    TFUtil::FeatureType type = static_cast<TFUtil::FeatureType>(type_arg.ints(0));

    // has_shape
    ndll_proto::Argument has_shape_arg = arg.extra_args(1);
    bool has_shape = has_shape_arg.bools(0);

    // shape
    ndll_proto::Argument shape_arg = arg.extra_args(2);
    std::vector<Index> shape{shape_arg.ints().begin(), shape_arg.ints().end()};

    // default value
    ndll_proto::Argument value_arg = arg.extra_args(3);
    TFUtil::Feature::Value val;
    switch (type) {
      case TFUtil::FeatureType::int64:
        val.int64 = value_arg.ints(0);
        break;
      case TFUtil::FeatureType::string:
        val.str = value_arg.strings(0);
        break;
      case TFUtil::FeatureType::float32:
        val.float32 = value_arg.floats(0);
        break;
      default:
        NDLL_FAIL("Unknown TFUtil::FeatureType value");
    }

    if (has_shape) {
      return Feature(shape, type, val);
    } else {
      return Feature(type, val);
    }
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
