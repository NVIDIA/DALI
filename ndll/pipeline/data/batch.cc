#include "ndll/pipeline/data/batch.h"

namespace ndll {

template <>
void BatchCopyHelper(const Batch<CPUBackend> &src, Batch<CPUBackend> *dst) {
  dst->set_type(src.type());
  dst->ResizeLike(src);
  std::memcpy(dst->raw_mutable_data(), src.raw_data(), src.nbytes());
}

} // namespace ndll
