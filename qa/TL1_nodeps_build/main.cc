#include <iostream>
#include "dali/kernels/alloc_type.h"
#include "dali/kernels/alloc.h"

int main(int argc, char **argv) {
  // Just using some dali kernel functions
  {
    auto mem = dali::kernels::memory::alloc_unique<float>(dali::kernels::AllocType::Host, 30);
  }

  {
    auto mem = dali::kernels::memory::alloc_unique<float>(dali::kernels::AllocType::GPU, 30);
  }

  return 0;
}
