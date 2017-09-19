#include "host_buffer.h"

#include "common.h"
#include "debug.h"

#include <cuda_runtime_api.h>

#include <cstring>

//
/// HostBuffer
//

HostBuffer::HostBuffer(unsigned int nSize): pData_(nullptr), nSize_(nSize) {
  if (nSize_ > 0) {
    CHECK_CUDA(cudaHostAlloc((void**)&pData_, nSize_, cudaHostAllocDefault));
  }
}

HostBuffer::~HostBuffer() {
  if (pData_) CHECK_CUDA(cudaFreeHost(pData_));
}

void
HostBuffer::resize(unsigned int nSize) {
  if (nSize > nSize_) {
    nSize_ = 0;
    CHECK_CUDA(cudaFreeHost(pData_));
    CHECK_CUDA(cudaHostAlloc((void**)&pData_, nSize, cudaHostAllocDefault));
    nSize_ = nSize;
  }
}

unsigned char* HostBuffer::data() {
  return pData_;
}

const unsigned char* HostBuffer::data() const {
  return pData_;
}

unsigned int HostBuffer::size() const {
  return nSize_;
}

//
/// HostBlocksDCT
//

HostBlocksDCT::HostBlocksDCT(unsigned int nWidth, unsigned int nHeight): owned_(true),
                                                                         nWidth_(nWidth)
                                                                       , nHeight_(nHeight)
                                                                       , pBlocks_(0)
                                                                       , nSize_(0)
{
  nSize_ = get_size(nWidth, nHeight);
  if (nSize_ > 0)
    CHECK_CUDA(cudaHostAlloc((void**)&pBlocks_, nSize_, cudaHostAllocDefault));
}

HostBlocksDCT::HostBlocksDCT(unsigned int nWidth, unsigned int nHeight,
    short *ptr, size_t size) : owned_(false), nWidth_(nWidth), nHeight_(nHeight),
                               pBlocks_(ptr), nSize_(size) {
  if (size != get_size(nWidth, nHeight)) {
    string err_str = string("Invalid argument to construct HostBlocksDCT. ") +
      string("Input size must be equal to `get_size(width, height)`");
    throw std::runtime_error(err_str);
  }
}
HostBlocksDCT::~HostBlocksDCT()
{
  if (owned_) cudaFreeHost(pBlocks_);
}

// HostBlocksDCT& HostBlocksDCT::operator=(const HostBlocksDCT &other) {
//   if (&other != this) {
//     // Resize the buffer if necessary
//     resize(other.nWidth_, other.nHeight_);

//     // Copy the data over
//     memcpy(pBlocks_, other.pBlocks_, get_size(nWidth_, nHeight_));        
//   }
//   return *this;
// }

void HostBlocksDCT::resize(unsigned int nWidth, unsigned int nHeight)
{
  size_t nSize = get_size(nWidth, nHeight);

  if (!owned_) {
    if (nSize != nSize_) {
      throw std::runtime_error("HostBlocksDCT does not own this buffer, cannot resize");
    }
    return;
  }
  
  // if memory is large enough
  if (nSize_ >= nSize) {
    nWidth_    = nWidth;
    nHeight_   = nHeight;
  } else {
    nSize_ = 0;
    nWidth_ = 0;
    nHeight_ = 0;
    CHECK_CUDA(cudaFreeHost(pBlocks_));
    CHECK_CUDA(cudaHostAlloc((void**)&pBlocks_, nSize, cudaHostAllocDefault));
    nSize_      = nSize;
    nWidth_     = nWidth;
    nHeight_    = nHeight;
  }
}

size_t
HostBlocksDCT::get_size(unsigned int nWidth, unsigned int nHeight)
{
  // 64 DCT coefficients in each block. This provides 128-bytes alignment of each row.
  unsigned int nLineStep = nWidth * 64 * sizeof(short);
    
  // Additional alignment of height needed because of how NPP handles subsampling.
  return nLineStep * DivUp(nHeight, 2) * 2;
}

short *
HostBlocksDCT::blockData()
{
  return pBlocks_;
}

const
short *
HostBlocksDCT::blockData()
  const
{
  return pBlocks_;
}

unsigned int
HostBlocksDCT::lineStep()
  const
{
  return nWidth_ * 64 * sizeof(short);
}
