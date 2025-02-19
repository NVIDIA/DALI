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

#include <gtest/gtest.h>
#include <stdexcept>
#include <system_error>
#define DALI_ALLOW_NEW_C_API
#include "dali/dali.h"
#include "dali/c_api_2/error_handling.h"
#include "dali/core/cuda_error.h"

namespace dali {

template <typename ExceptionType>
daliResult_t ThrowAndTranslate(ExceptionType &&ex) {
  DALI_PROLOG();
  throw std::forward<ExceptionType>(ex);
  DALI_EPILOG();
}

template <typename ExceptionType>
void CheckException(ExceptionType &&ex, daliResult_t expected_result) {
  std::string message(ex.what());
  daliResult_t ret = ThrowAndTranslate(std::forward<ExceptionType>(ex));
  EXPECT_EQ(ret, expected_result);
  EXPECT_EQ(daliGetLastError(), expected_result);
  EXPECT_EQ(daliGetLastErrorMessage(), message);
  std::cout << daliGetErrorName(ret) << " "
            << daliGetLastErrorMessage() << std::endl;
  daliClearLastError();
  EXPECT_EQ(daliGetLastError(), DALI_SUCCESS);
  EXPECT_STREQ(daliGetLastErrorMessage(), "");
}

TEST(CAPI2InternalTest, ErrorTranslation) {
  CheckException(std::runtime_error("Runtime Error"), DALI_ERROR_INVALID_OPERATION);
  CheckException(std::bad_alloc(), DALI_ERROR_OUT_OF_MEMORY);
  CheckException(CUDABadAlloc(), DALI_ERROR_OUT_OF_MEMORY);
  CheckException(std::logic_error("Logic dictates that it's an error"), DALI_ERROR_INTERNAL);
  CheckException(std::out_of_range("Bullet left the shooting range"), DALI_ERROR_OUT_OF_RANGE);
  CheckException(invalid_key("This key doesn't fit into the keyhole."), DALI_ERROR_INVALID_KEY);

  CheckException(std::system_error(std::make_error_code(std::errc::no_such_file_or_directory)),
                 DALI_ERROR_PATH_NOT_FOUND);
  CheckException(std::system_error(std::make_error_code(std::errc::no_such_device_or_address)),
                 DALI_ERROR_PATH_NOT_FOUND);

  CheckException(std::system_error(std::make_error_code(std::errc::no_space_on_device)),
                  DALI_ERROR_IO_ERROR);
  CheckException(std::system_error(
                      std::make_error_code(std::errc::inappropriate_io_control_operation)),
                 DALI_ERROR_IO_ERROR);
  CheckException(std::system_error(std::make_error_code(std::io_errc::stream)),
                 DALI_ERROR_IO_ERROR);

  CheckException(std::system_error(std::make_error_code(std::errc::not_enough_memory)),
                 DALI_ERROR_OUT_OF_MEMORY);

  CheckException(std::system_error(std::make_error_code(std::errc::bad_file_descriptor)),
                 DALI_ERROR_SYSTEM);
  CheckException(std::system_error(std::make_error_code(std::errc::too_many_files_open)),
                 DALI_ERROR_SYSTEM);
}

}  // namespace dali
