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

#ifndef DALI_OPERATORS_READER_PARSER_TFRECORD_PARSER_H_
#define DALI_OPERATORS_READER_PARSER_TFRECORD_PARSER_H_

#ifdef DALI_BUILD_PROTO3

#include <vector>
#include <string>
#include <exception>
#include <functional>

#include "dali/core/common.h"
#include "dali/pipeline/operator/argument.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/operators/reader/parser/parser.h"
#include "dali/operators/reader/parser/tf_feature.h"
#include "dali/operators/reader/parser/example.pb.h"

namespace dali {

class TFRecordParser : public Parser<Tensor<CPUBackend>> {
 public:
  using FeatureType = TFUtil::FeatureType;
  using Feature = TFUtil::Feature;

  explicit TFRecordParser(const OpSpec& spec) :
    Parser<Tensor<CPUBackend>>(spec) {
    feature_names_ = spec.GetRepeatedArgument<string>("feature_names");
    features_ = spec.GetRepeatedArgument<Feature>("features");
    DALI_ENFORCE(feature_names_.size() == features_.size(),
        "Number of features needs to match number of feature names.");
    DALI_ENFORCE(features_.size() > 0,
        "No features provided");
  }

  void Parse(const Tensor<CPUBackend>& data, SampleWorkspace* ws) override {
    tensorflow::Example example;

    uint64_t length;
    uint32_t crc;

    const uint8_t* raw_data = data.data<uint8_t>();

    std::memcpy(&length, raw_data, sizeof(length));

    // Omit length and crc
    raw_data = raw_data + sizeof(length) + sizeof(crc);
    DALI_ENFORCE(example.ParseFromArray(raw_data, length),
      make_string("Error while parsing TFRecord file: ", data.GetSourceInfo(),
                  " (raw data length: ", length, "bytes)."));

    for (size_t i = 0; i < features_.size(); ++i) {
      auto& output = ws->Output<CPUBackend>(i);
      Feature& f = features_[i];
      std::string& name = feature_names_[i];
      auto& feature = example.features().feature();
      auto it = feature.find(name);
      if (it == feature.end()) {
        output.Resize({0});
        // set type
        switch (f.GetType()) {
          case FeatureType::int64:
             output.Resize({0}, TypeTable::GetTypeInfo(DALI_INT64));
            break;
          case FeatureType::string:
             output.Resize({0}, TypeTable::GetTypeInfo(DALI_UINT8));
            break;
          case FeatureType::float32:
             output.Resize({0}, TypeTable::GetTypeInfo(DALI_FLOAT));
            break;
        }
        output.SetSourceInfo(data.GetSourceInfo());
        continue;
      }
      auto& encoded_feature = it->second;
      if (f.HasShape() && f.GetType() != FeatureType::string) {
        if (f.Shape().empty()) {
          output.Resize({1});
        } else {
          output.Resize(f.Shape());
        }
      }
      ssize_t number_of_elms = 0;
      switch (f.GetType()) {
        case FeatureType::int64:
          number_of_elms = encoded_feature.int64_list().value().size();
          if (!f.HasShape()) {
            output.Resize(InferShape(f, number_of_elms));
          }
          DALI_ENFORCE(number_of_elms <= output.size(), make_string("Output tensor shape is too "
                       "small: [", output.shape(), "]. Expected at least ", number_of_elms,
                       " elements."));
          std::memcpy(output.mutable_data<int64_t>(),
              encoded_feature.int64_list().value().data(),
              encoded_feature.int64_list().value().size()*sizeof(int64_t));
          break;
        case FeatureType::string:
          if (!f.HasShape() || volume(f.Shape()) > 1) {
            DALI_FAIL("Tensors of strings are not supported.");
          }
          output.Resize({static_cast<Index>(encoded_feature.bytes_list().value(0).size())});
          std::memcpy(output.mutable_data<uint8_t>(),
              encoded_feature.bytes_list().value(0).c_str(),
              encoded_feature.bytes_list().value(0).size()*sizeof(uint8_t));
          break;
        case FeatureType::float32:
          number_of_elms = encoded_feature.float_list().value().size();
          if (!f.HasShape()) {
            output.Resize(InferShape(f, number_of_elms));
          }
          DALI_ENFORCE(number_of_elms <= output.size(), make_string("Output tensor shape is too "
                       "small: [", output.shape(), "]. Expected at least ", number_of_elms,
                       " elements."));
          std::memcpy(output.mutable_data<float>(),
              encoded_feature.float_list().value().data(),
              number_of_elms * sizeof(float));
          break;
      }
      output.SetSourceInfo(data.GetSourceInfo());
    }
  }

 private:
  std::vector<std::string> feature_names_;
  std::vector<Feature> features_;

  std::vector<Index> InferShape(Feature& feature, size_t feature_size) {
    if (feature.HasPartialShape()) {
      auto partial_shape = feature.PartialShape();
      auto m = std::accumulate(
        partial_shape.begin(), partial_shape.end(), 1, std::multiplies<int>());
      DALI_ENFORCE(feature_size % m == 0, "Feature size not matching partial shape");
      partial_shape.insert(partial_shape.begin(), feature_size / m);
      return partial_shape;
    } else {
      return {static_cast<Index>(feature_size)};
    }
  }
};

};  // namespace dali

#endif  // DALI_BUILD_PROTO3

#endif  // DALI_OPERATORS_READER_PARSER_TFRECORD_PARSER_H_
