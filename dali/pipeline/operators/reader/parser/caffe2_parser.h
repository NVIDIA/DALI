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

#ifndef DALI_PIPELINE_OPERATORS_READER_PARSER_CAFFE2_PARSER_H_
#define DALI_PIPELINE_OPERATORS_READER_PARSER_CAFFE2_PARSER_H_

#include <string>

#include "dali/pipeline/operators/reader/parser/parser.h"
#include "dali/pipeline/operators/reader/parser/caffe2.pb.h"

namespace dali {

// From C2: caffe2/image/image_input.h
// SINGLE_LABEL: single integer label for multi-class classification
// MULTI_LABEL_SPARSE: sparse active label indices for multi-label classification
// MULTI_LABEL_DENSE: dense label embedding vector for label embedding regression
// MULTI_LABEL_WEIGHTED_SPARSE: sparse active label indices with per-label weights
// for multi-label classification
enum LabelType {
  SINGLE_LABEL = 0,
  MULTI_LABEL_SPARSE = 1,
  MULTI_LABEL_DENSE = 2,
  MULTI_LABEL_WEIGHTED_SPARSE = 3
};

// Extract the size of the data stored in a protobuf tensor
template <typename T>
int proto_data_size(const caffe2::TensorProto& proto) {
  DALI_FAIL("Base proto_data_size should never be called");
}

template <>
int proto_data_size<int>(const caffe2::TensorProto& proto) {
  return proto.int32_data_size();
}

template <>
int proto_data_size<float>(const caffe2::TensorProto& proto) {
  return proto.float_data_size();
}

// Extract a single value contained in a protobuf tensor
template <typename T>
T proto_get_data(const caffe2::TensorProto& proto, const size_t idx) {
  DALI_FAIL("Base proto_get_data should never be called");
}

template <>
int proto_get_data<int>(const caffe2::TensorProto& proto, const size_t idx) {
  return proto.int32_data(idx);
}

template <>
float proto_get_data<float>(const caffe2::TensorProto& proto, const size_t idx) {
  return proto.float_data(idx);
}

// Extract the data contained in a protobuf tensor
template <typename T>
void extract_data(const caffe2::TensorProto& proto,
                  Tensor<CPUBackend>& t) {
  DALI_FAIL("Base method should never be called");
}

template <>
void extract_data<int>(const caffe2::TensorProto& proto,
                       Tensor<CPUBackend>& t) {
  auto size = proto.int32_data_size();

  t.Resize({size});

  int* t_data = t.mutable_data<int>();
  for (auto i = 0; i < size; ++i) {
    t_data[i] = proto.int32_data(i);
  }
}

template <>
void extract_data<float>(const caffe2::TensorProto& proto,
                         Tensor<CPUBackend>& t) {
  auto size = proto.float_data_size();

  t.Resize({size});

  float* t_data = t.mutable_data<float>();
  for (int i = 0; i < size; ++i) {
    t_data[i] = proto.float_data(i);
  }
}

template <>
void extract_data<int64_t>(const caffe2::TensorProto& proto,
                           Tensor<CPUBackend>& t) {
  auto size = proto.int64_data_size();

  t.Resize({size});

  int64_t* t_data = t.mutable_data<int64_t>();
  for (auto i = 0; i < size; ++i) {
    t_data[i] = proto.int64_data(i);
  }
}

template <typename T>
void ParseLabels(const caffe2::TensorProtos& protos,
                 const LabelType label_type,
                 const int num_labels,
                 SampleWorkspace* ws) {
  auto& label_tensor = ws->Output<CPUBackend>(1);
  switch (label_type) {
    case SINGLE_LABEL: {
      // single element, from protos(1) to Output(1)
      // ensure we only have a single label in the proto
      DALI_ENFORCE(proto_data_size<T>(protos.protos(1)) == 1);

      extract_data<T>(protos.protos(1), ws->Output<CPUBackend>(1));
      break;
    }
    case MULTI_LABEL_SPARSE: {
      // multiple labels, all 1. in elements defined in protos(1)
      auto& label_tensor = ws->Output<CPUBackend>(1);
      label_tensor.Resize({num_labels});

      auto& label_indices = protos.protos(1);
      const int label_data_size = proto_data_size<T>(label_indices);

      T* label_tensor_data = label_tensor.mutable_data<T>();
      std::memset(label_tensor_data, 0, num_labels*sizeof(T));
      for (int i = 0; i < label_data_size; ++i) {
        label_tensor_data[static_cast<int>(proto_get_data<T>(label_indices, i))]
          = static_cast<T>(1);
      }
      break;
    }
    case MULTI_LABEL_DENSE: {
      // multiple elements, stored contiguously
      extract_data<T>(protos.protos(1), ws->Output<CPUBackend>(1));
      break;
    }
    case MULTI_LABEL_WEIGHTED_SPARSE: {
      // multiple elements with distinct weights
      // indices [int/float] in protos(1), weights [float] in protos(2)
      label_tensor.Resize({num_labels});

      auto& label_indices = protos.protos(1);
      auto& label_weights = protos.protos(2);
      const int label_size = proto_data_size<T>(label_indices);

      float* label_tensor_data = label_tensor.mutable_data<float>();
      std::memset(label_tensor_data, 0, num_labels*sizeof(float));
      for (int i = 0; i < label_size; ++i) {
        auto idx = static_cast<int>(proto_get_data<T>(label_indices, i));
        label_tensor_data[idx] = proto_get_data<float>(label_weights, i);
      }
      break;
    }
    default: {
      DALI_FAIL("Unsupported label type");
    }
  }
}

class Caffe2Parser : public Parser<Tensor<CPUBackend>> {
 public:
  explicit Caffe2Parser(const OpSpec& spec)
    : Parser(spec),
      additional_inputs_(spec.GetArgument<int>("additional_inputs")),
      label_type_(static_cast<LabelType>(spec.GetArgument<int>("label_type"))),
      num_labels_(spec.GetArgument<int>("num_labels")) {}

  void Parse(const Tensor<CPUBackend>& data, SampleWorkspace* ws) override {
    caffe2::TensorProtos protos;

    DALI_ENFORCE(protos.ParseFromArray(data.data<uint8_t>(), data.size()));

    auto& image = ws->Output<CPUBackend>(0);

    const caffe2::TensorProto& image_proto = protos.protos(0);
    const caffe2::TensorProto& label_proto = protos.protos(1);

    // copy image -- if type is string, image is encoded, if bytes, image isn't encoded
    if (image_proto.data_type() == caffe2::TensorProto::STRING) {
      const string& image_data = image_proto.string_data(0);
      const size_t image_bytes = image_data.size();

      image.Resize({(Index)image_bytes});
      std::memcpy(image.mutable_data<uint8_t>(), image_data.data(), image_bytes);
    } else if (image_proto.data_type() == caffe2::TensorProto::BYTE) {
      const int C = (image_proto.dims_size() == 3) ? image_proto.dims(2) : 1;
      const int H = image_proto.dims(0);
      const int W = image_proto.dims(1);

      image.Resize({H, W, C});
      std::memcpy(image.mutable_data<uint8_t>(),
                  image_proto.byte_data().data(),
                  image_proto.byte_data().size());
    }
    image.SetSourceInfo(data.GetSourceInfo());

    // Parse all label types
    auto label_data_type = label_proto.data_type();
    if (label_data_type == caffe2::TensorProto::FLOAT) {
      ParseLabels<float>(protos, label_type_, num_labels_, ws);
    } else if (label_data_type == caffe2::TensorProto::INT32) {
      ParseLabels<int>(protos, label_type_, num_labels_, ws);
    } else {
      DALI_FAIL("Unsupported label data type");
    }

    // handle any additional protos defined
    // additional outputs start at Output(2)
    auto additional_proto_start = (label_type_ == MULTI_LABEL_WEIGHTED_SPARSE) ? 3 : 2;
    auto additional_proto_end = additional_proto_start + additional_inputs_;
    int current_output_idx = 2;

    for (int i = additional_proto_start; i < additional_proto_end; ++i) {
      auto& additional_proto = protos.protos(i);
      auto& output_tensor = ws->Output<CPUBackend>(current_output_idx);

      switch (additional_proto.data_type()) {
       case caffe2::TensorProto::FLOAT:
        extract_data<float>(additional_proto, output_tensor);
        break;
       case caffe2::TensorProto::INT32:
        extract_data<int>(additional_proto, output_tensor);
        break;
       case caffe2::TensorProto::INT64:
        extract_data<int64_t>(additional_proto, output_tensor);
        break;
       default:
        DALI_FAIL("Unsupported data type in additional proto");
      }
      current_output_idx++;
    }

    // handle bounding box if needed
    // Final proto -> final output
    if (protos.protos_size() == additional_proto_end + 1) {
      auto& bbox_proto = protos.protos(additional_proto_end);

      DALI_ENFORCE(bbox_proto.data_type() == caffe2::TensorProto::INT32);
      DALI_ENFORCE(bbox_proto.int32_data_size() == 4);

      extract_data<int>(bbox_proto, ws->Output<CPUBackend>(current_output_idx));
    }
  }

 private:
  // Necessary for accounting purposes?
  int additional_inputs_;
  // Necessary for accounting purposes.
  LabelType label_type_;
  int num_labels_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_READER_PARSER_CAFFE2_PARSER_H_
