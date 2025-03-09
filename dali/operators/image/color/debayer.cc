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

#include "dali/operators/image/color/debayer.h"

namespace dali {

DALI_SCHEMA(experimental__Debayer)
    .DocStr(R"code(Performs image demosaicing/debayering.

Converts single-channel image to RGB using specified color filter array.

The supported input types are ``uint8_t`` and ``uint16_t``.
The input images must be 2D tensors (``HW``) or 3D tensors (``HWC``) where the number of channels is 1.
The operator supports sequence of images/video-like inputs (layout ``FHW``).
The output of the operator is always ``HWC`` (or ``FHWC`` for sequences).

For example, the following snippet presents debayering of batch of image sequences::

  def bayered_sequence(sample_info):
    # some actual source of video inputs with corresponding pattern
    # as opencv-style string
    video, bayer_pattern = get_sequence(sample_info)
    if bayer_pattern == "bggr":
        blue_position = [0, 0]
    elif bayer_pattern == "gbrg":
        blue_position = [0, 1]
    elif bayer_pattern == "grbg":
        blue_position = [1, 0]
    else:
        assert bayer_pattern == "rggb"
        blue_position = [1, 1]
    return video, np.array(blue_position, dtype=np.int32)

  @pipeline_def
  def debayer_pipeline():
    bayered_sequences, blue_positions = fn.external_source(
      source=bayered_sequence, batch=False, num_outputs=2,
      layout=["FHW", None])  # note the "FHW" layout, for plain images it would be "HW"
    debayered_sequences = fn.experimental.debayer(
      bayered_sequences.gpu(), blue_position=blue_positions, algorithm='bilinear_npp')
    return debayered_sequences

)code")
    .NumInput(1)
    .NumOutput(1)
    .AddArg(debayer::kBluePosArgName, R"code(The layout of color filter array/bayer tile.

A position of the blue value in the 2x2 bayer tile.
The supported values correspond to the following OpenCV bayer layouts:

* ``(0, 0)`` - ``BG``/``BGGR``
* ``(0, 1)`` - ``GB``/``GBRG``
* ``(1, 0)`` - ``GR``/``GRBG``
* ``(1, 1)`` - ``RG``/``RGGB``

The argument follows OpenCV's convention of referring to a 2x2 tile that starts
in the second row and column of the sensors' matrix.

For example, the ``(0, 0)``/``BG``/``BGGR`` corresponds to the following matrix of sensors:

.. list-table::
   :header-rows: 0

   * - R
     - G
     - R
     - G
     - R
   * - G
     - **B**
     - **G**
     - B
     - G
   * - R
     - **G**
     - **R**
     - G
     - R
   * - G
     - B
     - G
     - B
     - G
)code",
            DALI_INT_VEC, true, true)
    .AddArg(debayer::kAlgArgName,
            R"code(The algorithm to be used when inferring missing colours for any given pixel.
Different algorithms are supported on the GPU and CPU.

**GPU Algorithms:**

 - ``bilinear_npp`` bilinear interpolation with chroma correlation for green values.

**CPU Algorithms:**

 - ``bilinear_ocv`` bilinear interpolation.
 - ``edgeaware_ocv`` edge-aware interpolation.
 - ``vng_ocv`` Variable Number of Gradients (VNG) interpolation (only ``uint8_t`` supported).
 - ``gray_ocv`` converts the image to grayscale with bilinear interpolation.)code",
            DALI_STRING)
    .InputLayout(0, {"HW", "HWC", "FHW", "FHWC"})
    .AllowSequences();

}  // namespace dali
