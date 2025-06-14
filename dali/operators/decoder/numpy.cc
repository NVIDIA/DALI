// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string_view>

#include "dali/core/convert.h"
#include "dali/operators/decoder/numpy.h"

namespace dali {

numpy::HeaderData ParseHeader(const std::string_view data) {
  if (data.size() < 10 || data.substr(1, 5) != "NUMPY") {
    DALI_FAIL("Got bad magic string for numpy header: ", data.substr(1, 5));
  } else if (data[6] != 1) {
    DALI_FAIL("Unsupported numpy file version. Only major version 1 is supported.");
  }

  uint16_t header_len = 0;
  std::memcpy(&header_len, &data[8], 2);
  if (header_len + 10 > data.size()) {
    DALI_FAIL("Header length exceeds input size.");
  }

  numpy::HeaderData header;
  numpy::ParseHeaderContents(header, data.substr(10, header_len));
  header.data_offset = 10 + header_len;
  return header;
}

bool NumpyDecoder::SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) {
  const auto &input = ws.Input<CPUBackend>(0);

  headers_.clear();
  headers_.reserve(input.num_samples());
  for (int sampleIdx = 0; sampleIdx < input.num_samples(); sampleIdx++) {
    const auto sampleData = std::string_view(static_cast<const char *>(input.raw_tensor(sampleIdx)),
                                             volume(input.tensor_shape(sampleIdx)));
    DALI_ENFORCE(!sampleData.empty(), "Input sample is empty. Expected a non-empty NPY file.");
    headers_.emplace_back(ParseHeader(sampleData));
  }

  output_desc.resize(1);
  if (dtype_.has_value()) {
    output_desc[0].type = dtype_.value();
  } else {
    const auto firstDtype = headers_.front().type();
    DALI_ENFORCE(
        std::all_of(headers_.begin(), headers_.end(),
                    [&](const numpy::HeaderData &header) { return header.type() == firstDtype; }),
        "All samples in the batch must have the same data type, but got differing types");
    output_desc[0].type = firstDtype;
  }

  TensorListShape<-1> output_shape(headers_.size(), headers_.front().shape.sample_dim());
  for (int sampleIdx = 0; sampleIdx < input.num_samples(); ++sampleIdx) {
    const auto &header = headers_[sampleIdx];
    if (header.shape.sample_dim() != output_shape.sample_dim()) {
      DALI_FAIL(
          make_string("All samples in the batch must have the same number of dimensions, "
                      "but got differing sample dimensions: ",
                      output_shape.sample_dim(), " and ", header.shape.sample_dim()));
    }
    if (header.fortran_order) {
      // Fortran order means the shape is transposed, we need to reverse
      auto transposed_shape = TensorShape<-1>::empty_shape(output_shape.sample_dim());
      std::reverse_copy(header.shape.begin(), header.shape.end(), transposed_shape.begin());
      output_shape.set_tensor_shape(sampleIdx, transposed_shape);
    } else {
      output_shape.set_tensor_shape(sampleIdx, header.shape);
    }
  }
  output_desc[0].shape = output_shape;

  return true;
}

void RunDecoding(SampleView<CPUBackend> outputSample, ConstSampleView<CPUBackend> inputView,
                 const numpy::HeaderData &header) {
  Tensor<CPUBackend> transposeBuffer;
  if (header.fortran_order) {
    transposeBuffer.Resize(outputSample.shape(), inputView.type());
    auto transposeBufferView = SampleView<CPUBackend>(
        transposeBuffer.raw_mutable_data(), transposeBuffer.shape(), transposeBuffer.type());
    numpy::FromFortranOrder(transposeBufferView, inputView);
    inputView = ConstSampleView<CPUBackend>(transposeBuffer.raw_data(), transposeBuffer.shape(),
                                            transposeBuffer.type());
  }

  if (outputSample.type() != inputView.type()) {
    // If the types do not match, we need to convert the data
    TYPE_SWITCH(outputSample.type(), type2id, OType, NUMPY_ALLOWED_TYPES, (
      TYPE_SWITCH(inputView.type(), type2id, IType, NUMPY_ALLOWED_TYPES, (
        std::transform(inputView.data<IType>(),
          inputView.data<IType>() + volume(inputView.shape()),
          outputSample.mutable_data<OType>(), ConvertSat<OType, IType>);),
      DALI_FAIL(make_string("Unsupported input type: ", inputView.type())););),
    DALI_FAIL(make_string("Unsupported output type: ", outputSample.type())););
  } else {
    // If the types match, we can just copy the data
    auto *out_ptr = outputSample.raw_mutable_data();
    const auto *in_ptr = inputView.raw_data();
    std::memcpy(out_ptr, in_ptr, header.nbytes());
  }
}


void NumpyDecoder::RunImpl(Workspace &ws) {
  const auto &input = ws.Input<CPUBackend>(0);
  auto &output = ws.Output<CPUBackend>(0);
  auto &tPool = ws.GetThreadPool();

  for (int sampleIdx = 0; sampleIdx < input.num_samples(); ++sampleIdx) {
    tPool.AddWork([&, sampleIdx](int) {
      const numpy::HeaderData &header = headers_[sampleIdx];
      ConstSampleView<CPUBackend> inputView(input.raw_tensor(sampleIdx) + header.data_offset,
                                            header.shape, header.type());
      RunDecoding(output[sampleIdx], inputView, header);
    });
  }
  tPool.RunAll();
}

DALI_REGISTER_OPERATOR(decoders__Numpy, NumpyDecoder, CPU);

DALI_SCHEMA(decoders__Numpy)
    .DocStr(R"code(Decodes NumPy arrays from a serialized npy file.
The input should be a 1D uint8 tensor containing the binary data of the NumPy file.
All samples in the batch must have the same number of dimensions and data type (unless `dtype` is specified
which casts all samples in the batch to this dtype).
The output will be a tensor with the same shape and data type as the original NumPy array.

If the `dtype` argument is not specified, it will be inferred from the input data.
The operator supports both C-style (C-contiguous) and Fortran-style (Fortran-contiguous) arrays.
The operator does not support decoding of NumPy arrays with complex data types (e.g., structured arrays) and will raise an error
if the file is not `Format Version 1.0 <https://numpy.org/devdocs/reference/generated/numpy.lib.format.html#format-version-1-0>`_.
)code")
    .NumInput(1)
    .NumOutput(1)
    .InputDox(0, "data", "1D Tensor",
              R"code(Input that contains the binary data of the NumPy array.)code")
    .AddOptionalArg<DALIDataType>(
        "dtype",
        R"code(Data type of the output tensor. If not specified, it will be inferred from the input data.)code",
        nullptr);

}  // namespace dali
