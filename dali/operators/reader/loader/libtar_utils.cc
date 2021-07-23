#include <cstdlib>
#include <fcntl.h>
#include <future>
#include <algorithm>

#include "dali/operators/reader/loader/libtar_utils.h"
#include "dali/core/error_handling.h"

namespace dali {
namespace detail {

TarArchive::TarArchive(const string& filepath) {
  if (tar_open(&handle, filepath.c_str(), nullptr, O_RDONLY, 0, TAR_GNU)) {
    string error = "Could not open the tar archive at ";
    error += filepath;
    DALI_ERROR(error);
    handle = nullptr;
  }
}

TarArchive::TarArchive(TarArchive&& other) : handle(other.handle) {
  other.handle = nullptr;
}

TarArchive::~TarArchive() {
  TryClose();
}

bool TarArchive::Next() {
  if (CheckEnd()) {
    return;
  }

  if(tar_skip_regfile(handle)) {
    TryClose();
  }

  buffer_init = false;

  return CheckEnd();
}

inline bool TarArchive::CheckEnd() {
  return handle == nullptr;
}

void TarArchive::TryClose() {
  if (handle && tar_close(handle)) {
    string error = "Could not close the tar archive at ";
    error += handle->pathname;
    DALI_ERROR(error);
  }
  handle = nullptr;
}

string TarArchive::FileName() {
  return CheckEnd() ? "" : th_get_pathname(handle);
}

string TarArchive::FileRead(uint64 count) {
  if (CheckEnd()) {
    return "";
  }
  
  string out;
  BufferTryInit();
  while (buffer_size == T_BLOCKSIZE && count) {
    count -= BufferRead(out, count);
    if (buffer_offset == T_BLOCKSIZE) {
      BufferUpdate();
    }
  }

  return std::move(out);
}

bool TarArchive::Eof() {
  if (CheckEnd()) {
    return true;
  }

  BufferTryInit();
  return buffer_size == buffer_offset;
}

inline void TarArchive::BufferTryInit() {
  if (!buffer_init) {
    BufferUpdate();
    buffer_init = true;
  }
}

inline uint64 TarArchive::BufferRead(string& out, uint64 count) {
  count = std::min(count, (uint64)buffer_size - buffer_offset);
  out.append(buffer + buffer_offset, count);
  buffer_offset += count;
}

inline void TarArchive::BufferUpdate() {
  buffer_offset = 0;
  buffer_size = tar_block_read(handle, buffer);
}

}  // namespace detail
}  // namespace dali