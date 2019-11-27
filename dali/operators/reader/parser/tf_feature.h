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

#ifndef DALI_OPERATORS_READER_PARSER_TF_FEATURE_H_
#define DALI_OPERATORS_READER_PARSER_TF_FEATURE_H_

#ifdef DALI_BUILD_PROTO3

#include <string>
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/pipeline/proto/dali_proto_utils.h"

namespace dali {

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

  Feature() : has_shape_(false), type_(int64) {}

  Feature(std::vector<Index> shape, FeatureType type, Value val) {
    has_shape_ = true;
    shape_ = std::move(shape);
    type_ = type;
    val_ = std::move(val);
  }

  Feature(FeatureType type, Value val, std::vector<Index> partial_shape) {
    has_shape_ = false;
    has_partial_shape_ = true;
    partial_shape_ = std::move(partial_shape);
    type_ = type;
    val_ = std::move(val);
  }

  Feature(FeatureType type, Value val) {
    has_shape_ = false;
    type_ = type;
    val_ = std::move(val);
  }

  FeatureType GetType() const { return type_; }
  bool HasShape() const { return has_shape_; }
  bool HasPartialShape() const { return has_partial_shape_; }
  const std::vector<Index>& Shape() const {
    return shape_;
  }
  std::vector<Index> PartialShape() const {
    return partial_shape_;
  }
  const Value GetValue() const { return val_; }

  template<typename T>
  T GetDefaultValue() {
    if (std::is_same<T, std::string>::value()) {
      DALI_ENFORCE(GetType() == FeatureType::string,
          "Requested invalid type from the Feature");
      return val_.str;
    } else if (std::is_same<T, int64_t>::value()) {
      DALI_ENFORCE(GetType() == FeatureType::int64,
          "Requested invalid type from the Feature");
      return val_.int64;
    } else if (std::is_same<T, float>::value()) {
      DALI_ENFORCE(GetType() == FeatureType::float32,
          "Requested invalid type from the Feature");
      return val_.float32;
    } else {
      DALI_FAIL("Requested invalid type from the feature");
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

  DaliProtoPriv * SerializeToProtobuf(DaliProtoPriv *arg) const {
    arg->set_type(SerializeType());
    arg->set_is_vector(false);

    // set the datatype of the record
    auto type_arg = arg->add_extra_args();
    type_arg.set_name("type");
    dali::SerializeToProtobuf(static_cast<int>(GetType()), &type_arg);

    // has_shape
    auto has_shape_arg = arg->add_extra_args();
    has_shape_arg.set_name("has_shape");
    dali::SerializeToProtobuf(HasShape(), &has_shape_arg);

    // set the shape
    auto shape_arg = arg->add_extra_args();
    shape_arg.set_name("shape");
    shape_arg.set_is_vector(false);
    auto& shape = Shape();
    for (size_t i = 0; i < shape.size(); ++i) {
      dali::SerializeToProtobuf(shape[i], &shape_arg);
    }

    // set the default value
    auto default_arg = arg->add_extra_args();
    default_arg.set_name("default_value");
    switch (GetType()) {
      case TFUtil::FeatureType::int64:
        dali::SerializeToProtobuf(GetValue().int64, &default_arg);
        break;
      case TFUtil::FeatureType::string:
        dali::SerializeToProtobuf(GetValue().str, &default_arg);
        break;
      case TFUtil::FeatureType::float32:
        dali::SerializeToProtobuf(GetValue().float32, &default_arg);
        break;
      default:
        DALI_FAIL("Unknown TFUtil::FeatureType value");
    }
    return arg;
  }

  static Feature DeserializeFromProtobuf(const DaliProtoPriv& arg) {
    // type argument
    DaliProtoPriv type_arg = arg.extra_args(0);
    TFUtil::FeatureType type = static_cast<TFUtil::FeatureType>(type_arg.ints(0));

    // has_shape
    DaliProtoPriv has_shape_arg = arg.extra_args(1);
    bool has_shape = has_shape_arg.bools(0);

    // shape
    DaliProtoPriv shape_arg = arg.extra_args(2);
    const auto shape = shape_arg.ints();

    // default value
    DaliProtoPriv value_arg = arg.extra_args(3);
    TFUtil::Feature::Value val = {};
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
        DALI_FAIL("Unknown TFUtil::FeatureType value");
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
  bool has_partial_shape_ = false;
  std::vector<Index> partial_shape_;
};

}  // namespace TFUtil

}  // namespace dali

#endif  // DALI_BUILD_PROTO3

#endif  // DALI_OPERATORS_READER_PARSER_TF_FEATURE_H_
