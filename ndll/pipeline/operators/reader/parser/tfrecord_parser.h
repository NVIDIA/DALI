// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_READER_PARSER_TFRECORD_PARSER_H_
#define NDLL_PIPELINE_OPERATORS_READER_PARSER_TFRECORD_PARSER_H_

#if NDLL_USE_PROTOBUF

#include <vector>
#include <string>

#include "ndll/pipeline/operators/reader/parser/parser.h"
#include "ndll/pipeline/operators/reader/parser/example.pb.h"
#include "ndll/common.h"

namespace ndll {

class TFRecordParser : public Parser {
 public:
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


   private:
    bool has_shape_;
    std::vector<Index> shape_;
    FeatureType type_;
    Value val_;
  };
  explicit TFRecordParser(const OpSpec& spec) :
    Parser(spec) {}

  void Parse(const uint8_t* data, const size_t size, SampleWorkspace* ws) override {
    tensorflow::Example example;
  }
};

};  // namespace ndll

#endif  // NDLL_USE_PROTOBUF
#endif  // NDLL_PIPELINE_OPERATORS_READER_PARSER_TFRECORD_PARSER_H_
