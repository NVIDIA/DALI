// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

DALI_SCHEMA(experimental__Inflate)
    .DocStr(R"code(Inflates/decompresses the input using specified decompression algorithm.

The input must be a 1D tensor of bytes (uint8). Passing the ``shape`` and ``dtype`` of the
decompressed samples is required.

Each input sample can either be a single compressed chunk or consist of multiple
compressed chunks that have the same shape and type when inflated, so that they can be
be merged into a single tensor where the leftmost extent of the tensor corresponds
to the number of the chunks.

If the sample is comprised of multiple chunks, the ``chunks_offsets`` or ``chunks_sizes``
must be specified. In that case, the ``shape`` must describe the shape of a single inflated
(output) chunk. The number of the chunks will automatically be added as the leftmost extent
to the output tensors.

For example, the following snippet presents decompression of a video-like sequences.
Each video sequence was deflated by, first, compressing each frame separately and then
concatenating compressed frames from the corresponding sequences.::

  @pipeline_def
  def inflate_sequence_pipeline():
    compressed_seq, uncompressed_hwc_shape, compressed_chunks_sizes = fn.external_source(...)
    sequences = fn.experimental.inflate(
        compressed_seq.gpu(),
        chunks_sizes=compressed_chunks_sizes,  # refers to sizes in ``compressed_seq``
        shape=uncompressed_hwc_shape,
        layout="HWC",
        sequence_extent="F")
    return sequences

)code")
    .NumInput(1)
    .NumOutput(1)
    .AddArg(inflate::shapeArgName, "The shape of the output (inflated) chunk.", DALI_INT_VEC, true)
    .AddOptionalTypeArg(inflate::dTypeArgName, "The output (inflated) data type.", DALI_NO_TYPE)
    .AddOptionalArg(inflate::offsetArgName, R"code("A list of offsets within the input sample
describing where the consecutive chunks begin.

If the ``chunks_sizes`` is not specified, it is assumed that the chunks are densely packed
in the input tensor and the last chunk ends with the sample's end.)code",
                    std::vector<int>{}, true)
    .AddOptionalArg(inflate::sizeArgName,
                    R"code("A list of sizes of corresponding input chunks.

If the ``chunks_offsets`` is not specified, it is assumed that the chunks are densely packed
in the input tensor and the first chunk starts at the beginning of the sample.)code",
                    std::vector<int>{}, true)
    .AddOptionalArg(inflate::algArgName, R"code(Algorithm to be used to decode the data.

Currently only ``LZ4`` is supported.)code",
                    "LZ4")
    .AddOptionalArg(inflate::layoutArgName, R"code(Layout of the output (inflated) chunk.

If the samples consist of multiple chunks, additionally, the ``sequence_extent`` extent
will be added to the beginning of the specified layout.)code",
                    TensorLayout(""))
    .AddOptionalArg(inflate::sequenceLayoutArgName, R"code(The name for the sequence extent.

If the samples consist of multiple chunks, the extra sequence extent will be added as the leftmost
extent to the output tensor. By default it is assumed to be video frames,
hence the default ``F`` extent.

The value is ignored if the ``layout`` is not specified or the input is not a sequence
( neither ``chunks_offsets`` nor ``chunks_sizes`` is specified).
)code",
                    TensorLayout("F"));

}  // namespace dali
