// Source: filename is based off idea behind Caffe file of the same
// name. Contents have no relation unless otherwise specified.
#ifndef NDLL_COMMON_H_
#define NDLL_COMMON_H_

#include <cstdint>

#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <cuda_fp16.h> // for __half & related methods

namespace ndll {

// Using declaration for common types
using std::array;
using std::cout;
using std::endl;
using std::vector;
using std::shared_ptr;
using std::string;

// Common types
typedef uint8_t uint8;

// Only supported on the GPU
typedef __half float16;


// Source: based on Caffe macro of the same name. Impl uses 'delete'
// instead of just making these functions private
//
// Helper to delete copy constructor & copy-assignment operator
#define DISABLE_COPY_MOVE_ASSIGN(name)          \
  name(const name&) = delete;                   \
  name& operator=(const name&) = delete;        \
  name(name&&) = delete;                        \
  name& operator=(name&&) = delete
  



} // namespace ndll

#endif // NDLL_COMMON_H_
