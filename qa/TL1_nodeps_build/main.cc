#include <iostream>
#include "dali/kernels/alloc_type.h"
#include "dali/core/mm/memory.h"

int main(int argc, char **argv) {
  // Just using some dali kernel functions
  {
    auto mem = dali::kernels::memory::alloc_unique<float>(dali::kernels::AllocType::Host, 30);
  }

  return 0;
}
