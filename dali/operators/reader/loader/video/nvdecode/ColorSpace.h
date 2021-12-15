#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

void Nv12ToColor32(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgra, int nBgraPitch, int nWidth, int nHeight, int iMatrix);
