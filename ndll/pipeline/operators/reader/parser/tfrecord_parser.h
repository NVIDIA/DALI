// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_READER_PARSER_TFRECORD_PARSER_H_
#define NDLL_PIPELINE_OPERATORS_READER_PARSER_TFRECORD_PARSER_H_

#ifdef NDLL_BUILD_PROTO3

#include <vector>
#include <string>
#include <exception>

#include "ndll/common.h"
#include "ndll/pipeline/operators/argument.h"
#include "ndll/pipeline/operators/op_spec.h"
#include "ndll/pipeline/operators/reader/parser/parser.h"
#include "ndll/pipeline/operators/reader/parser/tf_feature.h"
#include "ndll/pipeline/operators/reader/parser/example.pb.h"

namespace ndll {

class TFRecordParser : public Parser {
 public:
  using FeatureType = TFUtil::FeatureType;
  using Feature = TFUtil::Feature;

  explicit TFRecordParser(const OpSpec& spec) :
    Parser(spec) {
    feature_names_ = spec.GetRepeatedArgument<string>("feature_names");
    features_ = spec.GetRepeatedArgument<Feature>("features");
    NDLL_ENFORCE(feature_names_.size() == features_.size(),
        "Number of features needs to match number of feature names.");
    NDLL_ENFORCE(features_.size() > 0,
        "No features provided");
  }

  void Parse(const uint8_t* data, const size_t size, SampleWorkspace* ws) override {
    tensorflow::Example example;

    uint64_t length;
    uint32_t crc;

    std::memcpy(&length, data, sizeof(length));

    // Omit length and crc
    data = data + sizeof(length) + sizeof(crc);
    try {
      NDLL_ENFORCE(example.ParseFromArray(data, length),
          "Error in parsing - invalid TFRecord file!");
    } catch(std::exception& e) {
      std::string str = "Error while parsing TFRecord: " + std::string(e.what());
      throw std::runtime_error(str);
    }

    for (size_t i = 0; i < features_.size(); ++i) {
      auto* output = ws->Output<CPUBackend>(i);
      Feature& f = features_[i];
      std::string& name = feature_names_[i];
      auto& encoded_feature = example.features().feature().at(name);
      if (f.HasShape() && f.GetType() != FeatureType::string) {
        if (f.Shape().empty()) {
        output->Resize({1});
        } else {
          output->Resize(f.Shape());
        }
      }
      switch (f.GetType()) {
        case FeatureType::int64:
          if (!f.HasShape()) {
            output->Resize({encoded_feature.int64_list().value().size()});
          }
          std::memcpy(output->mutable_data<int64_t>(),
              encoded_feature.int64_list().value().data(),
              encoded_feature.int64_list().value().size()*sizeof(int64_t));
          break;
        case FeatureType::string:
          if (!f.HasShape() || Product(f.Shape()) > 1) {
            NDLL_FAIL("Tensors of strings are not supported.");
          }
          output->Resize({static_cast<Index>(encoded_feature.bytes_list().value(0).size())});
          std::memcpy(output->mutable_data<uint8_t>(),
              encoded_feature.bytes_list().value(0).c_str(),
              encoded_feature.bytes_list().value(0).size()*sizeof(uint8_t));
          break;
        case FeatureType::float32:
          if (!f.HasShape()) {
            output->Resize({encoded_feature.float_list().value().size()});
          }
          std::memcpy(output->mutable_data<float>(),
              encoded_feature.float_list().value().data(),
              encoded_feature.float_list().value().size()*sizeof(float));
          break;
      }
    }
  }

 private:
  std::vector<std::string> feature_names_;
  std::vector<Feature> features_;
};

};  // namespace ndll

#endif  // NDLL_BUILD_PROTO3

#endif  // NDLL_PIPELINE_OPERATORS_READER_PARSER_TFRECORD_PARSER_H_
