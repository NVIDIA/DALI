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

#ifndef DALI_UTIL_FITS_H_
#define DALI_UTIL_FITS_H_

#include <fitsio.h>
#include <set>
#include <string>
#include "dali/core/common.h"
#include "dali/core/static_switch.h"
#include "dali/core/stream.h"
#include "dali/core/tensor_shape.h"
#include "dali/core/unique_handle.h"
#include "dali/kernels/transpose/transpose.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/sample_view.h"
#include "dali/pipeline/data/tensor.h"


namespace dali {
namespace fits {

const std::set<DALIDataType> supportedTypes = {DALI_UINT8,   DALI_UINT16, DALI_UINT32, DALI_UINT64,
                                               DALI_INT8,    DALI_INT16,  DALI_INT32,  DALI_INT64,
                                               DALI_FLOAT16, DALI_FLOAT,  DALI_FLOAT64};

inline std::string SupportedTypesListGen() {
  std::stringstream out;
  for (auto &dtype : supportedTypes) {
    out << dtype << ", ";
  }
  std::string out_str = out.str();
  return out_str.substr(0, out_str.size() - 2 * (supportedTypes.size() > 0));
}

class DLL_PUBLIC HeaderData {
 public:
  TensorShape<> shape;
  int hdu_type;
  int datatype_code;
  const TypeInfo *type_info = nullptr;
  bool compressed = false;

  DALIDataType type() const;

  size_t size() const;

  size_t nbytes() const;
};

DLL_PUBLIC void ParseHeader(HeaderData &parsed_header, fitsfile *src);


class DLL_PUBLIC FitsHandle : public UniqueHandle<fitsfile *, FitsHandle> {
 public:
  DALI_INHERIT_UNIQUE_HANDLE(fitsfile *, FitsHandle)
  constexpr FitsHandle() = default;

  /** @brief Opens the FITS file with fits_open_file*/
  static FitsHandle OpenFile(const char *path, int mode) {
    int status = 0;
    fitsfile *ff = nullptr;

    fits_open_file(&ff, path, mode, &status);
    DALI_ENFORCE(status == 0,
                 make_string("Failed to open a file: ", path, " Make sure it exists!"));

    return FitsHandle(ff);
  }


  /** @brief Calls fits_close_file on the file handle */
  static void DestroyHandle(fitsfile *ff) {
    int status = 0;
    fits_close_file(ff, &status);
    DALI_ENFORCE(status == 0,
                 make_string("Failed while executing fits_close_file! Status code: ", status));
  }
};


DLL_PUBLIC std::string GetFitsErrorMessage(int status);

DLL_PUBLIC void HandleFitsError(int status);

/** @brief Wrapper that automatically handles cfitsio error checking.*/
DLL_PUBLIC void FITS_CALL(int status);

}  // namespace fits
}  // namespace dali

#endif  // DALI_UTIL_FITS_H_
