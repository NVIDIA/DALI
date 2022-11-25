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

The input images must be 2D tensors (``HW``) or 3D tensors (``HWC``) where the number of channels is 1.
The supported input types are ``uint8_t`` and ``uint16_t``.

)code")
    .NumInput(1)
    .NumOutput(1)
    .AddArg(debayer::bluePosArgName, R"code(The layout of color filter array/bayer tile.

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
    .AddOptionalArg(
        debayer::algArgName,
        R"code(The algorithm to be used when inferring missing colours for any given pixel.
Currently only ``bilinear_npp`` is supported.

* The ``bilinear_npp`` algorithm uses bilinear interpolation to infer red and blue values.
  For green values a bilinear interpolation with chroma correlation is used as explained in
  `NPP documentation <https://docs.nvidia.com/cuda/npp/group__image__color__debayer.html>`_.)code",
        "bilinear_npp")
    .InputLayout(0, {"HW", "HWC"})
    .AllowSequences();

}  // namespace dali
