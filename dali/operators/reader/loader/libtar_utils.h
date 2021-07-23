#ifndef DALI_OPERATORS_READER_LOADER_LIBTAR_UTILS_H_
#define DALI_OPERATORS_READER_LOADER_LIBTAR_UTILS_H_

#include <libtar.h>

#include <iterator>
#include <cstddef>

#include "dali/core/common.h"

namespace dali {
namespace detail {

class TarArchive {
 public:
  explicit TarArchive(const string& filepath);
  TarArchive(TarArchive&&);
  ~TarArchive();
  
  bool Next();
  bool CheckEnd();

  string FileName();
  string FileRead(uint64 count);
  bool Eof();

 private:
  void TryClose();
  TAR* handle;

  void BufferTryInit();
  uint64 BufferRead(string&, uint64);
  void BufferUpdate();

  char buffer[T_BLOCKSIZE];
  uint8 buffer_offset;
  uint8 buffer_size;
  bool buffer_init;
};

}  // namespace detail
}  // namespace dali
#endif /* DALI_OPERATORS_READER_LOADER_LIBTAR_UTILS_H_ */
