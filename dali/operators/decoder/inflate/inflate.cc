// Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/decoder/inflate/inflate.h"

namespace dali {

DALI_SCHEMA(decoders__Inflate)
    .DocStr(R"code(Decompresses the input using specified decompression algorithm.

The input must be a 1D tensor of bytes (uint8). Passing the `shape` and `dtype` of the
decompressed samples is required.

Each input sample can either be a single compressed chunk or consist of multiple
compressed chunks that have the same shape and type when inflated, so that they can be
be merged into a single tensor where the outermost extent of the tensor corresponds
to the number of the chunks.

If the sample is comprised of multiple chunks, the `chunk_offsets` or `chunk_sizes`
must be specified. In that case, the `shape` must describe the shape of a single inflated
(output) chunk. The number of the chunks will automatically be added as the outermost extent
to the output tensors.

For example, the following snippet presents decompression of a video-like sequences.
Each video sequence was deflated by, first, compressing each frame separately and then
concatenating compressed frames from the corresponding sequences.::

  @pipeline_def
  def inflate_sequence_pipeline():
    compres_seq, uncompres_hwc_shape, compres_chunk_sizes = fn.external_source(...)
    sequences = fn.decoders.inflate(
        compres_seq.gpu(),
        chunk_sizes=compres_chunk_sizes,  # refers to sizes in ``compres_seq``
        shape=uncompres_hwc_shape,
        layout="HWC",
        sequence_axis_name="F")
    return sequences

)code")
    .NumInput(1)
    .NumOutput(1)
    .AddArg(inflate::shapeArgName, "The shape of the output (inflated) chunk.", DALI_INT_VEC, true)
    .AddOptionalTypeArg(inflate::dTypeArgName, "The output (inflated) data type.", DALI_UINT8)
    .AddOptionalArg<std::vector<int>>(inflate::offsetArgName,
                                      R"code(A list of offsets within the input sample
describing where the consecutive chunks begin.

If the `chunk_sizes` is not specified, it is assumed that the chunks are densely packed
in the input tensor and the last chunk ends with the sample's end.)code",
                                      nullptr, true)
    .AddOptionalArg<std::vector<int>>(inflate::sizeArgName,
                                      R"code(A list of sizes of corresponding input chunks.

If the `chunk_offsets` is not specified, it is assumed that the chunks are densely packed
in the input tensor and the first chunk starts at the beginning of the sample.)code",
                                      nullptr, true)
    .AddOptionalArg(inflate::algArgName, R"code(Algorithm to be used to decode the data.

Currently only ``LZ4`` is supported.)code",
                    "LZ4")
    .AddOptionalArg(inflate::layoutArgName,
                    R"code(Layout of the output (inflated) chunk.

If the samples consist of multiple chunks, additionally, the `sequence_axis_name` extent
will be added to the beginning of the specified layout.)code",
                    TensorLayout(""))
    .AddOptionalArg(inflate::sequenceLayoutArgName, R"code(The name for the sequence axis.

If the samples consist of multiple chunks, an extra outer dimension will be added to
the output tensor. By default, it is assumed to be video frames, hence the default label 'F'

The value is ignored if the `layout` is not specified or the input is not a sequence
( neither `chunk_offsets` nor `chunk_sizes` is specified).
)code",
                    TensorLayout("F"));

DALI_SCHEMA(experimental__Inflate)
    .AddParent("decoders__Inflate")
    .NumInput(1)
    .NumOutput(1)
    .Deprecate("2.0", "decoders__Inflate")
    .MakeDocHidden();

}  // namespace dali
