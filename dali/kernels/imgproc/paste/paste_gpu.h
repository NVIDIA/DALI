// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// THIS WHOLE CODE IS A COPY-PASTE OF multiply-add kernel

#ifndef DALI_KERNELS_IMGPROC_PASTE_PASTE_GPU_H_
#define DALI_KERNELS_IMGPROC_PASTE_PASTE_GPU_H_

#include <vector>
#include <unordered_set>
#include <set>
#include <map>
#include <tuple>
#include "dali/core/span.h"
#include "dali/core/convert.h"
#include "dali/core/geom/box.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/imgproc/roi.h"
#include "dali/kernels/common/block_setup.h"
#include "dali/kernels/imgproc/paste/paste_gpu_input.h"

namespace dali {
namespace kernels {
namespace paste {

template<class InputType, int ndims>
struct PatchDesc {
  const InputType *in;
  int in_sample_idx;
  ivec<ndims> patch_start, patch_end, in_anchor, in_pitch;
};

template <class OutputType, class InputType, int ndims>
struct SampleDescriptorGPU {
  OutputType *out;
  int patch_start_idx;
  ivec<ndims> patch_counts, out_pitch;
};

// Could not fill this in Setup - there are no pointers yet
template <class OutputType, class InputType, int ndims>
void FillPointers(span<SampleDescriptorGPU<OutputType, InputType, ndims - 1>> descs,
                  span<PatchDesc<InputType, ndims - 1>> patches,
                  const OutListGPU<OutputType, ndims> &out,
                  const InListGPU<InputType, ndims> &in) {
  int batch_size = descs.size();

  for (int i = 0; i < batch_size; i++) {
    descs[i].out = out.data[i];
  }

  for (auto &p : patches) {
    p.in = p.in_sample_idx < 0 ? nullptr : in.data[p.in_sample_idx];
  }
}

/**
 * Note: Since the operation we perform is channel-agnostic (it is performed in the same
 * way for every channel), SampleDescriptor assumes, that sample is channel-agnostic. Therefore it
 * needs to be flattened
 * @see FlattenChannels
 */
template <class OutputType, class InputType, int ndims>
void CreateSampleDescriptors(
    vector<SampleDescriptorGPU<OutputType, InputType, ndims - 1>> &out_descs,
    vector<PatchDesc<InputType, ndims - 1>> &out_patches,
    const TensorListShape<ndims> &in_shape,
    span<paste::MultiPasteSampleInput<ndims - 1>> samples) {
  static_assert(ndims == 3, "Only 2D data with channels supported");

  int batch_size = samples.size();

  out_descs.resize(batch_size);
  out_patches.clear();

  for (int out_idx = 0; out_idx < batch_size; out_idx++) {
    const auto &sample = samples[out_idx];
      const int channels = sample.channels;
      int n = sample.inputs.size();

      // Get all significant points on x and y axis
      // Those will be the starts and ends of the patches
      int NO_DATA = 0;
      std::map<int, int> x_points;
      std::map<int, int> y_points;
      x_points[0] = NO_DATA;
      x_points[sample.out_size[1]] = NO_DATA;
      y_points[0] = NO_DATA;
      y_points[sample.out_size[0]] = NO_DATA;

      for (int i = 0; i < n; i++) {
        x_points[sample.inputs[i].out_anchor[1]] = NO_DATA;
        x_points[sample.inputs[i].out_anchor[1] + sample.inputs[i].size[1]] = NO_DATA;
        y_points[sample.inputs[i].out_anchor[0]] = NO_DATA;
        y_points[sample.inputs[i].out_anchor[0] + sample.inputs[i].size[0]] = NO_DATA;
      }

      // When we know how many of those points there are, we know how big our patch patch is
      int x_patch_cnt = static_cast<int>(x_points.size()) - 1;
      int y_patch_cnt = static_cast<int>(y_points.size()) - 1;

      // Now lets fill forward and backward mapping of those significant points to patch indices
      vector<int> scaled_x_to_x;
      for (auto &x_point : x_points) {
        x_point.second = scaled_x_to_x.size();
        scaled_x_to_x.push_back(x_point.first);
      }
      vector<int> scaled_y_to_y;
      for (auto &y_point : y_points) {
        y_point.second = scaled_y_to_y.size();
        scaled_y_to_y.push_back(y_point.first);
      }

      // We create events that will fire when sweeping
      vector<vector<std::tuple<int, int, int>>> y_starting(y_patch_cnt + 1);
      vector<vector<std::tuple<int, int, int>>> y_ending(y_patch_cnt + 1);
      for (int i = 0; i < n; i++) {
        y_starting[y_points[sample.inputs[i].out_anchor[0]]].emplace_back(
                i, x_points[sample.inputs[i].out_anchor[1]],
                x_points[sample.inputs[i].out_anchor[1] + sample.inputs[i].size[1]]);
        y_ending[y_points[sample.inputs[i].out_anchor[0] + sample.inputs[i].size[0]]].emplace_back(
                i, x_points[sample.inputs[i].out_anchor[1]],
                x_points[sample.inputs[i].out_anchor[1] + sample.inputs[i].size[1]]);
      }
      y_starting[0].emplace_back(-1, 0, x_patch_cnt);
      y_ending[y_patch_cnt].emplace_back(-1, 0, x_patch_cnt);

      // Filling sample
      int prev_patch_count = out_patches.size();
      auto &out_sample = out_descs[out_idx];
      out_sample.patch_start_idx = prev_patch_count;
      out_sample.patch_counts[1] = x_patch_cnt;
      out_sample.patch_counts[0] = y_patch_cnt;
      out_sample.out_pitch[1] = sample.out_size[1] * channels;

      // And now the sweeping itself
      out_patches.resize(prev_patch_count + x_patch_cnt * y_patch_cnt);
      vector<std::unordered_set<int>> starting(x_patch_cnt + 1);
      vector<std::unordered_set<int>> ending(x_patch_cnt + 1);
      std::set<int> open_pastes;
      for (int y = 0; y < y_patch_cnt; y++) {
        // Add open and close events on x axis for regions with given y start coordinate
        for (auto &i : y_starting[y]) {
          starting[std::get<1>(i)].insert(std::get<0>(i));
          ending[std::get<2>(i)].insert(std::get<0>(i));
        }
        // Now sweep through x
        for (int x = 0; x < x_patch_cnt; x++) {
          // Open regions starting here
          for (int i : starting[x]) {
            open_pastes.insert(i);
          }

          // Take top most region
          int max_paste = *(--open_pastes.end());
          auto& patch = out_patches[prev_patch_count + y * x_patch_cnt + x];


          // And fill the patch
          patch.patch_start[0] = scaled_y_to_y[y];
          patch.patch_start[1] = scaled_x_to_x[x];
          patch.patch_end[0] = scaled_y_to_y[y + 1];
          patch.patch_end[1] = scaled_x_to_x[x + 1];
          patch.in = nullptr;  // to be filled later
          patch.in_sample_idx = max_paste == -1 ? -1 : sample.inputs[max_paste].in_idx;
          patch.in_pitch[0] = 0;
          patch.in_pitch[1] = max_paste == -1 ? -1 : in_shape[sample.inputs[max_paste].in_idx][1];

          if (max_paste != -1) {
            patch.in_anchor[0] = sample.inputs[max_paste].in_anchor[0] +
                              patch.patch_start[0] - sample.inputs[max_paste].out_anchor[0];
            patch.in_anchor[1] = sample.inputs[max_paste].in_anchor[1] +
                              patch.patch_start[1] - sample.inputs[max_paste].out_anchor[1];
          }
          patch.patch_start[1] *= channels;
          patch.patch_end[1] *= channels;
          patch.in_pitch[1] *= channels;
          patch.in_anchor[1] *= channels;

          // Now remove regions that end here
          for (int i : ending[x + 1]) {
            open_pastes.erase(i);
          }
        }
        // And remove start/events for regions whose y ends here
        for (auto &i : y_ending[y + 1]) {
          starting[std::get<1>(i)].erase(std::get<0>(i));
          ending[std::get<2>(i)].erase(std::get<0>(i));
        }
      }
  }
}



template <class OutputType, class InputType, int ndims>
__global__ void
PasteKernel(const SampleDescriptorGPU<OutputType, InputType, ndims> *samples,
            const PatchDesc<InputType, ndims> *patches,
            const BlockDesc<ndims> *blocks) {
  static_assert(ndims == 2, "Function requires 2 dimensions in the input");
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];
  const PatchDesc<InputType, ndims> *my_patches = patches + sample.patch_start_idx;


  auto *__restrict__ out = sample.out;

  int patch_y = 0;
  int min_patch_x = 0;
  while (block.start.x + static_cast<int>(threadIdx.x) >= my_patches[min_patch_x].patch_end[1]) {
    min_patch_x++;
  }

  for (int y = block.start.y + static_cast<int>(threadIdx.y);
       y < block.end.y; y += static_cast<int>(blockDim.y)) {
    while (y >= my_patches[patch_y * sample.patch_counts[1]].patch_end[0]) patch_y++;
    int patch_x = min_patch_x;
    for (int x = block.start.x + static_cast<int>(threadIdx.x);
         x < block.end.x; x += static_cast<int>(blockDim.x)) {
      while (x >= my_patches[patch_y * sample.patch_counts[1] + patch_x].patch_end[1]) patch_x++;
      const PatchDesc<InputType, ndims> *patch =
              &my_patches[patch_y * sample.patch_counts[1] + patch_x];
      if (patch->in == nullptr) {
          out[y * sample.out_pitch[1] + x] = 0;
      } else {
        out[y * sample.out_pitch[1] + x] = ConvertSat<OutputType>(
                patch->in[(y - patch->patch_start[0] + patch->in_anchor[0]) * patch->in_pitch[1]
                         + (x - patch->patch_start[1] + patch->in_anchor[1])]);
      }
    }
  }
}

}  // namespace paste

template <typename OutputType, typename InputType, int ndims>
class PasteGPU {
 private:
  static constexpr size_t spatial_dims = ndims - 1;
  using BlockDesc = kernels::BlockDesc<spatial_dims>;
  using SampleDesc = paste::SampleDescriptorGPU<OutputType, InputType, spatial_dims>;
  using PatchDesc = paste::PatchDesc<InputType, spatial_dims>;

  std::vector<SampleDesc> sample_descriptors_;
  std::vector<PatchDesc> patch_descriptors_;

 public:
  BlockSetup<spatial_dims, -1 /* No channel dimension, only spatial */> block_setup_;

  KernelRequirements Setup(
      KernelContext &context,
      span<paste::MultiPasteSampleInput<spatial_dims>> samples,
      const TensorListShape<ndims> &out_shape,
      const TensorListShape<ndims> &in_shape) {
    paste::CreateSampleDescriptors(sample_descriptors_, patch_descriptors_, in_shape, samples);

    KernelRequirements req;
    ScratchpadEstimator se;
    // merge width with channels
    auto flattened_shape = collapse_dim(out_shape, spatial_dims - 1);
    block_setup_.SetupBlocks(flattened_shape, true);
    se.add<mm::memory_kind::device, SampleDesc>(sample_descriptors_.size());
    se.add<mm::memory_kind::device, PatchDesc>(patch_descriptors_.size());
    se.add<mm::memory_kind::device, BlockDesc>(block_setup_.Blocks().size());
    req.output_shapes = { out_shape };
    req.scratch_sizes = se.sizes;
    return req;
  }


  void Run(
      KernelContext &context,
      const OutListGPU<OutputType, ndims> &out,
      const InListGPU<InputType, ndims> &in) {
    paste::FillPointers(make_span(sample_descriptors_), make_span(patch_descriptors_), out, in);

    SampleDesc *samples_gpu;
    PatchDesc  *patches_gpu;
    BlockDesc *blocks_gpu;

    std::tie(samples_gpu, patches_gpu, blocks_gpu) = context.scratchpad->ToContiguousGPU(
        context.gpu.stream, sample_descriptors_, patch_descriptors_, block_setup_.Blocks());

    dim3 patch_dim = block_setup_.GridDim();
    dim3 block_dim = block_setup_.BlockDim();
    auto stream = context.gpu.stream;

    paste::PasteKernel<<<patch_dim, block_dim, 0, stream>>>(
        samples_gpu, patches_gpu, blocks_gpu);
    CUDA_CALL(cudaGetLastError());
  }
};

}  // namespace kernels
}  // namespace dali
#endif  // DALI_KERNELS_IMGPROC_PASTE_PASTE_GPU_H_
