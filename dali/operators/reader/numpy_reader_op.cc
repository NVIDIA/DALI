// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <string>


#include "dali/pipeline/data/views.h"
#include "dali/kernels/common/transpose.h"
#include "dali/core/static_switch.h"
#include "dali/operators/reader/numpy_reader_op.h"

namespace dali {

#define NUMPY_ALLOWED_TYPES \
  (bool, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, float16, \
  double)


TensorListShape<> NumpyReader::GetSliceArg(ArgumentWorkspace &ws, const char *name) {
  if (spec_.HasArgument(name)) {
    return uniform_list_shape(batch_size_, spec_.GetRepeatedArgument<int>(name));
  } else if (spec_.HasTensorArgument(name)) {
    auto &t = ws.ArgumentInput(name);
    DALI_ENFORCE(static_cast<int>(t.size()) == batch_size_
                 && t.shape().sample_dim() == 1
                 && is_uniform(t.shape()),
                 "Shape must be a list of 1D tensors of equal length");
    auto tlv = view<const int>(t);
    TensorListShape<> tls;
    tls.resize(batch_size_, t.shape()[0][0]);
    TensorShape<> ts;
    ts.resize(tls.sample_dim());
    for (int i = 0; i < batch_size_; i++) {
      auto tv = tlv[i];
      for (int j = 0; j < ts.shape[0]; j++)
        ts[j] = tv.data[j];
      tls.set_tensor_shape(i, ts);
    }
    return tls;
  } else {
    // create an list of empty shapes
    return {};
  }
}

void NumpyReader::Prefetch() {
  // We actually prepare the next batch
  TimeRange tr("NumpyReader::Prefetch #" + to_string(curr_batch_producer_), TimeRange::kRed);
  auto &curr_batch = prefetched_batch_queue_[curr_batch_producer_];
  curr_batch.reserve(batch_size_);
  curr_batch.clear();

  if (slab_shapes_.empty() || slab_anchors_.empty()) {
    loader_->SetSlabParameters({}, {});
    for (int i = 0; i < batch_size_; ++i) {
      curr_batch.push_back(loader_->ReadOne(i == 0));
    }
  } else {
    for (int i = 0; i < batch_size_; ++i) {
      loader_->SetSlabParameters(slab_anchors_[i], slab_shapes_[i]);
      curr_batch.push_back(loader_->ReadOne(i == 0));
    }
    DALI_FAIL("Sliced reads not supported yet!");
  }
}

// Run the operator
void NumpyReader::Run(HostWorkspace &ws) {
  // Get the arguments
  slab_anchors_ = GetSliceArg(ws, "anchor");
  slab_shapes_ = GetSliceArg(ws, "shape");

  // If necessary start prefetching thread and wait for a consumable batch
  StartPrefetchThread();
  ConsumerWait();

  // consume batch
  TimeRange tr("NumpyReader::Run #" + to_string(curr_batch_consumer_), TimeRange::kViolet);

  // This is synchronous call for CPU Backend
  Operator<CPUBackend>::Run(ws);

  // Notify that we have consumed whole batch
  ConsumerAdvanceQueue();
}

// Actual Read implementation
void NumpyReader::RunImpl(SampleWorkspace &ws) {
  // get the ws index
  const int idx = ws.data_idx();

  // get sample
  const auto& imfile = GetSample(idx);

  // copy from raw_data -> outputs directly
  auto& image_output = ws.Output<CPUBackend>(0);

  if (imfile.meta == "transpose:false") {
    // just copy the tensor over
    CopyHelper(image_output, imfile.image);
  } else {
    // here we need to transpose the data
    TransposeHelper(image_output, imfile.image);
  }
  image_output.SetSourceInfo(imfile.image.GetSourceInfo());
}

// data copy helpers
void NumpyReader::CopyHelper(Tensor<CPUBackend>& output, const Tensor<CPUBackend>& input) {
  output.Resize(input.shape(), input.type());
  std::memcpy(output.raw_mutable_data(), input.raw_data(), input.nbytes());
}

void NumpyReader::TransposeHelper(Tensor<CPUBackend>& output, const Tensor<CPUBackend>& input) {
  auto& in_shape = input.shape();
  auto& type = input.type();
  int n_dims = in_shape.size();
  std::vector<int> perm(n_dims);
  std::vector<int64_t> out_shape(n_dims);
  for (int i = 0; i < n_dims; ++i) {
    perm[i] = n_dims - i - 1;
    out_shape[i] = in_shape[perm[i]];
  }
  output.Resize(out_shape, type);
  auto input_type = type.id();
  TensorShape<> in_ts(in_shape.begin(), in_shape.end());
  TensorShape<> out_ts(out_shape.begin(), out_shape.end());
  TYPE_SWITCH(input_type, type2id, InputType, NUMPY_ALLOWED_TYPES, (
    kernels::Transpose(
      TensorView<StorageCPU, InputType>{output.mutable_data<InputType>(), out_ts},
      TensorView<StorageCPU, const InputType>{input.data<InputType>(), in_ts},
      make_cspan(perm));), DALI_FAIL("Input type not supported."));
}

DALI_REGISTER_OPERATOR(NumpyReader, NumpyReader, CPU);

DALI_SCHEMA(NumpyReader)
  .DocStr("Read Numpy arrays from a directory")
  .NumInput(0)
  .NumOutput(1)  // (Arrays)
  .AddArg("file_root",
      R"code(Path to a directory containing data files.
`NumpyReader` supports flat directory structure. `file_root` directory should contain
directories with numpy files in them.)code",
      DALI_STRING)
  .AddOptionalArg("file_filter",
      R"code(If specified, the string will be interpreted as glob string to filter
the list of files in the sub-directories of `file_root`.)code", "*.npy")
  .AddOptionalArg("shuffle_after_epoch",
      R"code(If true, reader shuffles whole dataset after each epoch. It is exclusive with
`stick_to_shard` and `random_shuffle`.)code",
      false)
  .AddOptionalArg<int>("anchor", R"code(Specifies the anchor for sliced reads.\n
If no anchor is specified, the whole file is read.)code",
      std::vector<int>(), true)
  .AddOptionalArg<int>("shape", R"code(Specifies the shape of the slice for sliced reads.\n
If no shape is specified, the whole file is read.)code",
      std::vector<int>(), true)
  .AddParent("LoaderBase");

}  // namespace dali
