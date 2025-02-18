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

#include <stdexcept>
#include <system_error>
#include <string>
#include <utility>
#include "dali/c_api_2/error_handling.h"
#include "dali/core/error_handling.h"
#include "dali/core/cuda_error.h"

namespace dali::c_api {

struct ErrorInfo {
  inline ErrorInfo() {
    message.reserve(1024);  // should help with properly reporting OOMs
  }
  daliResult_t  result = DALI_SUCCESS;
  std::string   message;
};

namespace {
thread_local ErrorInfo g_daliLastError;

// A workaround for category comparison acros shared object boundary
inline bool CategoryEqual(const std::error_category &c1, const std::error_category &c2) {
  if (c1 == c2)  // we might get false negatives here...
    return true;
  // ...so we compare names - it's not foolproof, but should work for builtin categories,
  // which is all we're intersted in.
  const char *n1 = c1.name();
  const char *n2 = c2.name();
  return n1 == n2 || !strcmp(n1, n2);
}

}  // namespace

struct ErrorDesc {
  const char *name, *description;
};

ErrorDesc GetErrorDesc(daliResult_t result) {
  #define RESULT_DESC(name, desc) case DALI_##name: return { "DALI_" #name, desc }
  #define ERROR_DESC(name, desc) RESULT_DESC(ERROR_##name, desc)

  switch (result) {
    RESULT_DESC(SUCCESS, "The operation was successful.");
    RESULT_DESC(NO_DATA, "The operation was successful, but didn't return any data.");
    RESULT_DESC(NOT_READY, "The query succeeded, but the operation queried is still pending.");
    ERROR_DESC(INVALID_HANDLE, "The operation received an invalid DALI handle.");
    ERROR_DESC(INVALID_ARGUMENT, "An invalid argument was specified.");
    ERROR_DESC(INVALID_TYPE, "An argument of invalid type encountered.");
    ERROR_DESC(INVALID_OPERATION, "An invalid operation was requested.");
    ERROR_DESC(OUT_OF_RANGE, "An argument is out of valid range.");
    ERROR_DESC(INVALID_KEY, "The operation received an invalid dictionary key.");

    ERROR_DESC(SYSTEM, "An operating system routine failed.");
    ERROR_DESC(PATH_NOT_FOUND, "A non-existent or non-accessible file path was encountered.");
    ERROR_DESC(IO_ERROR, "An I/O operation failed");
    ERROR_DESC(OUT_OF_MEMORY, "Cannot allocate memory");
    ERROR_DESC(INTERNAL, "An internal error occurred");
    ERROR_DESC(UNLOADING, "DALI is unloading - either daliShutdown was called or "
                          "the process is shutting down.");
    ERROR_DESC(CUDA_ERROR, "A CUDA call has failed.");
    default:
      return { "<invalid>", "<invalid>" };
  }
}

daliResult_t SetLastError(daliResult_t result, const char *message) {
  g_daliLastError.result  = result;
  g_daliLastError.message = message;
  return result;
}

daliResult_t HandleError(std::exception_ptr ex) {
  try {
    std::rethrow_exception(std::move(ex));
  } catch (dali::c_api::InvalidHandle &e) {
    return SetLastError(DALI_ERROR_INVALID_HANDLE, e.what());
  } catch (std::invalid_argument &e) {
    return SetLastError(DALI_ERROR_INVALID_ARGUMENT, e.what());
  } catch (dali::CUDAError &e) {
    if (e.is_rt_api()) {
      if (e.rt_error() == cudaErrorNotReady)
        return SetLastError(DALI_NOT_READY, e.what());
    } else if (e.is_drv_api()) {
      if (e.drv_error() == CUDA_ERROR_NOT_READY)
        return SetLastError(DALI_NOT_READY, e.what());
    }
    return SetLastError(DALI_ERROR_CUDA_ERROR, e.what());
  } catch (dali::CUDABadAlloc &e) {
    return SetLastError(DALI_ERROR_OUT_OF_MEMORY, e.what());
  } catch (std::bad_alloc &e) {
    return SetLastError(DALI_ERROR_OUT_OF_MEMORY, e.what());
  } catch (dali::invalid_key &e) {
    return SetLastError(DALI_ERROR_INVALID_KEY, e.what());
  } catch (std::out_of_range &e) {
    return SetLastError(DALI_ERROR_OUT_OF_RANGE, e.what());
  } catch (std::system_error &e) {
    // compare by name to work around DLL issues
    if (CategoryEqual(e.code().category(), std::generic_category())) {
      daliResult_t result = [&]() {
        switch (static_cast<std::errc>(e.code().value())) {
        case std::errc::no_such_file_or_directory:
        case std::errc::no_such_device:
        case std::errc::no_such_device_or_address:
          return DALI_ERROR_PATH_NOT_FOUND;
        case std::errc::not_enough_memory:
          return DALI_ERROR_OUT_OF_MEMORY;
        case std::errc::timed_out:
          return DALI_ERROR_TIMEOUT;
        case std::errc::address_family_not_supported:
        case std::errc::address_in_use:
        case std::errc::address_not_available:
        case std::errc::already_connected:
        case std::errc::broken_pipe:
        case std::errc::connection_aborted:
        case std::errc::connection_already_in_progress:
        case std::errc::connection_refused:
        case std::errc::connection_reset:
        case std::errc::device_or_resource_busy:
        case std::errc::directory_not_empty:
        case std::errc::file_exists:
        case std::errc::file_too_large:
        case std::errc::filename_too_long:
        case std::errc::host_unreachable:
        case std::errc::inappropriate_io_control_operation:
        case std::errc::io_error:
        case std::errc::is_a_directory:
        case std::errc::message_size:
        case std::errc::network_down:
        case std::errc::network_reset:
        case std::errc::network_unreachable:
        case std::errc::no_buffer_space:
        case std::errc::no_message:
        case std::errc::no_space_on_device:
        case std::errc::not_a_directory:
        case std::errc::not_a_socket:
        case std::errc::read_only_file_system:
          return DALI_ERROR_IO_ERROR;
        default:
          return DALI_ERROR_SYSTEM;
        }
      }();
      return SetLastError(result, e.what());
    } else if (CategoryEqual(e.code().category(), std::iostream_category())) {
      return SetLastError(DALI_ERROR_IO_ERROR, e.what());
    } else {
      return SetLastError(DALI_ERROR_SYSTEM, e.what());
    }
  } catch (std::runtime_error &e) {
    return SetLastError(DALI_ERROR_INVALID_OPERATION, e.what());
  } catch (std::exception &e) {
    return SetLastError(DALI_ERROR_INTERNAL, e.what());
  } catch (const char *e) {  // handle strings thrown as exceptions
    return SetLastError(DALI_ERROR_INTERNAL, e);
  } catch (const std::string &e) {  // handle strings thrown as exceptions
    return SetLastError(DALI_ERROR_INTERNAL, e.c_str());
  } catch (...) {
    return SetLastError(DALI_ERROR_INTERNAL, "<unknown error>");
  }
}

}  // namespace dali::c_api

using namespace dali;  // NOLINT

daliResult_t daliGetLastError() {
  return c_api::g_daliLastError.result;
}

const char *daliGetLastErrorMessage() {
  return c_api::g_daliLastError.message.c_str();
}

void daliClearLastError() {
  c_api::g_daliLastError = {};
}

const char *daliGetErrorName(daliResult_t result) {
  return c_api::GetErrorDesc(result).name;
}

const char *daliGetErrorDescription(daliResult_t result) {
  return c_api::GetErrorDesc(result).description;
}
