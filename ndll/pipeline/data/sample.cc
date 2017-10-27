#include "ndll/pipeline/data/sample.h"

#include <cstring>

#include "ndll/pipeline/data/backend.h"

namespace ndll {

template <>
void Sample<CPUBackend>::Copy(const Sample<CPUBackend> &other, cudaStream_t stream) {
  this->set_type(other.type());
  this->ResizeLike(other);
  std::memcpy(this->raw_mutable_data(), other.raw_data(),  other.nbytes());
}

template <>
void Sample<GPUBackend>::Copy(const Sample<GPUBackend> &other, cudaStream_t stream) {
  this->set_type(other.type());
  this->ResizeLike(other);
  MemCopy(this->raw_mutable_data(), other.raw_data(), other.nbytes(), stream);
}

} // namespace ndll
