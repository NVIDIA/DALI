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

#include "dali/operators/sequence/per_frame.h"

namespace dali {

DALI_SCHEMA(PerFrame)
    .DocStr(R"code(Marks the input tensor as a sequence.

The operator modifies the layout string of the input data to indicate that
the batch contains sequences.
Only the layout is affected, while the data stays untouched.

The operator can be used to feed per-frame tensor arguments when procesing sequences.
For example, the following snippet shows how to apply ``gaussian_blur`` to a batch of sequences,
so that a different ``sigma`` is used for each frame in each sequence::

  @pipeline_def
  def random_per_frame_blur():
    video, _ = fn.readers.video_resize(sequence_length=50, ...)
    sigma = fn.random.uniform(range=[0.5, 5], shape=(50,))
    blurred = fn.gaussian_blur(video, sigma=fn.per_frame(sigma))
    return blurred

Note that the outermost dimension of each tensor from a batch specified as per-frame argument
must match the number of frames in the corresponding sequence processed by a given operator.
For instance, in the presented example, every sequence in ``video`` batch has 50 frames,
thus the shape of ``sigma`` is ``(50,)``.

Please consult documentation of a given argument of a sequence processing operator
to find out if it supports per-frame input.

If the input passed to ``per-frame`` operator has no layout,
a new layout is set, that starts with ``F`` and is padded with ``*`` to match
dimensionality of the input. Otherwise, depending on the ``replace`` flag,
the operator either checks if the first character of the layout is equal to ``F``
or replaces the character with ``F``.
)code")
    .NumInput(1)
    .NumOutput(1)
    .PassThrough({{0, 0}})
    .AddOptionalArg<bool>("replace",
                          R"code(Controls handling of the input with already specified layout.
If set to False, the operator errors-out if the first character of the layout is not ``F``.
If set to True, the first character of the layout is replaced with ``F``.)code",
                          false);


DALI_REGISTER_OPERATOR(PerFrame, PerFrame<CPUBackend>, CPU);
DALI_REGISTER_OPERATOR(PerFrame, PerFrame<GPUBackend>, GPU);

}  // namespace dali
