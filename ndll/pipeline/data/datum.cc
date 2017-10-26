#include "ndll/pipeline/data/datum.h"

#include <cstring>

#include "ndll/pipeline/data/backend.h"

namespace ndll {

template <>
void Datum<CPUBackend>::Copy(const Datum<CPUBackend> &other, cudaStream_t stream) {
  this->set_type(other.type());
  this->ResizeLike(other);
  std::memcpy(this->raw_mutable_data(), other.raw_data(),  other.nbytes());
}

template <>
void Datum<GPUBackend>::Copy(const Datum<GPUBackend> &other, cudaStream_t stream) {
  this->set_type(other.type());
  this->ResizeLike(other);
  MemCopy(this->raw_mutable_data(), other.raw_data(), other.nbytes(), stream);
}

} // namespace ndll
