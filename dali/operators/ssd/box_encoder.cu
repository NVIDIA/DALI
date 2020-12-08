// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/ssd/box_encoder.cuh"
#include <cuda.h>
#include <vector>
#include <utility>

namespace dali {
__host__ __device__ inline float4 ToCenterWidthHeight(const float4 &box) {
    return {
      0.5f * (box.x + box.z),
      0.5f * (box.y + box.w),
      box.z - box.x,
      box.w - box.y};
}

void BoxEncoder<GPUBackend>::PrepareAnchors(const vector<float> &anchors) {
  DALI_ENFORCE(
    (anchors.size() % BoundingBox::size) == 0,
    "Anchors size must be divisible by 4, actual value = " + std::to_string(anchors.size()));

  anchors_count_ = anchors.size() / BoundingBox::size;
  anchors_.Resize({anchors_count_, static_cast<int64_t>(BoundingBox::size)});
  anchors_as_center_wh_.Resize({anchors_count_, static_cast<int64_t>(BoundingBox::size)});

  auto anchors_data_cpu = reinterpret_cast<const float4 *>(anchors.data());

  vector<float4> anchors_as_center_wh(anchors_count_);
  for (unsigned int anchor = 0; anchor < anchors_count_; ++anchor)
    anchors_as_center_wh[anchor] = ToCenterWidthHeight(anchors_data_cpu[anchor]);

  auto anchors_data = anchors_.mutable_data<float>();
  auto anchors_as_center_wh_data = anchors_as_center_wh_.mutable_data<float>();
  MemCopy(anchors_data, anchors.data(), anchors_count_ * BoundingBox::size * sizeof(float));
  MemCopy(
    anchors_as_center_wh_data,
    anchors_as_center_wh.data(),
    anchors_count_ * BoundingBox::size * sizeof(float));
}

__device__ __forceinline__ float CalculateIou(const float4 &b1, const float4 &b2) {
  float l = cuda_max(b1.x, b2.x);
  float t = cuda_max(b1.y, b2.y);
  float r = cuda_min(b1.z, b2.z);
  float b = cuda_min(b1.w, b2.w);
  float first = cuda_max(r - l, 0.0f);
  float second = cuda_max(b - t, 0.0f);
  volatile float intersection = first * second;
  volatile float area1 = (b1.w - b1.y) * (b1.z - b1.x);
  volatile float area2 = (b2.w - b2.y) * (b2.z - b2.x);

  return intersection / (area1 + area2 - intersection);
}

__device__ inline void FindBestMatch(const int N, volatile float *vals, volatile int *idx) {
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      if (vals[threadIdx.x] <= vals[threadIdx.x + stride]) {
        if (vals[threadIdx.x] == vals[threadIdx.x + stride]) {
          idx[threadIdx.x] = cuda_max(idx[threadIdx.x], idx[threadIdx.x + stride]);
        } else {
          vals[threadIdx.x] = vals[threadIdx.x + stride];
          idx[threadIdx.x] = idx[threadIdx.x + stride];
        }
      }
    }
    __syncthreads();
  }
}

__device__ float4 MatchOffsets(
  float4 box, float4 anchor, const float *means, const float *stds, float scale) {
  box.x *= scale; box.y *= scale; box.z *= scale; box.w *= scale;
  anchor.x *= scale; anchor.y *= scale; anchor.z *= scale; anchor.w *= scale;

  float x = ((box.x - anchor.x) / anchor.z - means[0]) / stds[0];
  float y = ((box.y - anchor.y) / anchor.w - means[1]) / stds[1];
  float z = (log(box.z / anchor.z) - means[2]) / stds[2];
  float w = (log(box.w / anchor.w) - means[3]) / stds[3];

  return {x, y, z, w};
}

__device__ void WriteMatchesToOutput(
  unsigned int anchors_count, float criteria, int *labels_out, const int *labels_in,
  float4 *boxes_out, const float4 *boxes_in,
  volatile int *best_box_idx, volatile float *best_box_iou, bool offset,
  const float* means, const float* stds, float scale, const float4 *anchors_as_cwh) {
  for (unsigned int anchor = threadIdx.x; anchor < anchors_count; anchor += blockDim.x) {
    if (best_box_iou[anchor] > criteria) {
      int box_idx = best_box_idx[anchor];
      labels_out[anchor] = labels_in[box_idx];
      float4 box = boxes_in[box_idx];

      if (!offset)
        boxes_out[anchor] = ToCenterWidthHeight(box);
      else
        boxes_out[anchor] = MatchOffsets(
          ToCenterWidthHeight(box), anchors_as_cwh[anchor], means, stds, scale);
    }
  }
}

__device__ void MatchBoxWithAnchors(
  const float4 &box, const int box_idx, unsigned int anchors_count, const float4 *anchors,
  volatile int *best_anchor_idx_tmp, volatile float *best_anchor_iou_tmp,
  volatile int *best_box_idx, volatile float *best_box_iou) {
  float best_anchor_iou = -1.0f;
  int best_anchor_idx = -1;

  for (unsigned int anchor = threadIdx.x; anchor < anchors_count; anchor += blockDim.x) {
    float new_val = CalculateIou(box, anchors[anchor]);

    if (new_val >= best_anchor_iou) {
        best_anchor_iou = new_val;
        best_anchor_idx = anchor;
    }

    if (new_val >= best_box_iou[anchor]) {
        best_box_iou[anchor] = new_val;
        best_box_idx[anchor] = box_idx;
    }
  }

  best_anchor_iou_tmp[threadIdx.x] = best_anchor_iou;
  best_anchor_idx_tmp[threadIdx.x] = best_anchor_idx;
}

template <int BLOCK_SIZE>
__global__ void Encode(
  const float4 *boxes_in, const int *labels_in, const int *offsets, const int anchors_count,
  const float4 *anchors, const float criteria, float4 *boxes_out,  int *labels_out,
  int *box_idx_buffer, float *box_iou_buffer, bool offset, const float* means,
  const float* stds, float scale, const float4 *anchors_as_cwh) {
  const int sample = blockIdx.x;

  // Remark: This algorithm is very fragile to floating point arithmetic effects.
  // For now, excessive use of volatile in this code,
  // makes it conform to reference solution in terms of resulting encoding.

  __shared__ volatile int best_anchor_idx_tmp[BLOCK_SIZE];
  __shared__ volatile float best_anchor_iou_tmp[BLOCK_SIZE];

  volatile int *best_box_idx = box_idx_buffer + sample * anchors_count;
  volatile float *best_box_iou = box_iou_buffer + sample * anchors_count;

  int box_idx = 0;
  for (int box_global_idx = offsets[sample]; box_global_idx < offsets[sample+1]; ++box_global_idx) {
    MatchBoxWithAnchors(
      boxes_in[box_global_idx],
      box_idx,
      anchors_count,
      anchors,
      best_anchor_idx_tmp,
      best_anchor_iou_tmp,
      best_box_idx,
      best_box_iou);

    __syncthreads();

    FindBestMatch(blockDim.x, best_anchor_iou_tmp, best_anchor_idx_tmp);
    __syncthreads();

    if (threadIdx.x == 0) {
      int idx = best_anchor_idx_tmp[0];
      best_box_idx[idx] = box_idx;
      best_box_iou[idx] = 2.f;
    }
    __syncthreads();

    box_idx++;
  }
  __syncthreads();

  WriteMatchesToOutput(
    anchors_count,
    criteria,
    labels_out + sample * anchors_count,
    labels_in + offsets[sample],
    boxes_out + sample * anchors_count,
    boxes_in + offsets[sample],
    best_box_idx,
    best_box_iou,
    offset,
    means,
    stds,
    scale,
    anchors_as_cwh);
}

std::pair<int *, float *> BoxEncoder<GPUBackend>::ClearBuffers(const cudaStream_t &stream) {
  auto best_box_idx_data = best_box_idx_.mutable_data<int>();
  auto best_box_iou_data = best_box_iou_.mutable_data<float>();

  CUDA_CALL(cudaMemsetAsync(
    best_box_idx_data, 0, curr_batch_size_ * anchors_count_ * sizeof(int), stream));
  CUDA_CALL(cudaMemsetAsync(
    best_box_iou_data, 0, curr_batch_size_ * anchors_count_ * sizeof(float), stream));

  return {best_box_idx_data, best_box_iou_data};
}

void BoxEncoder<GPUBackend>::WriteAnchorsToOutput(
  float4 *boxes_out_data, int *labels_out_data,  const cudaStream_t &stream) {
  CUDA_CALL(cudaMemsetAsync(
    labels_out_data,
    0,
    curr_batch_size_ * anchors_count_ * sizeof(int), stream));

  for (int sample = 0; sample < curr_batch_size_; ++sample)
    MemCopy(
      boxes_out_data + sample * anchors_count_,
      anchors_as_center_wh_.data<float>(),
      anchors_count_ * BoundingBox::size * sizeof(float),
      stream);
}

void BoxEncoder<GPUBackend>::ClearOutput(
  float4 *boxes_out_data, int *labels_out_data,  const cudaStream_t &stream) {
  CUDA_CALL(cudaMemsetAsync(
    labels_out_data,
    0,
    curr_batch_size_ * anchors_count_ * sizeof(int),
    stream));

  for (int sample = 0; sample < curr_batch_size_; ++sample)
    CUDA_CALL(cudaMemsetAsync(
      boxes_out_data + sample * anchors_count_,
      0,
      anchors_count_ * BoundingBox::size * sizeof(float),
      stream));
}

std::pair<TensorListShape<>, TensorListShape<>>
BoxEncoder<GPUBackend>::CalculateDims(
  const TensorList<GPUBackend> &boxes_input) {
  TensorListShape<> boxes_output_shape(boxes_input.ntensor(), kBoxesOutputDim);
  TensorListShape<> labels_output_shape(boxes_input.ntensor(), kLabelsOutputDim);

  for (size_t i = 0; i < boxes_input.ntensor(); i++) {
    boxes_output_shape.set_tensor_shape(i,
        {anchors_count_, static_cast<int64_t>(BoundingBox::size)});
    labels_output_shape.set_tensor_shape(i, {anchors_count_});
  }

  return {boxes_output_shape, labels_output_shape};
}

int *BoxEncoder<GPUBackend>::CalculateBoxesOffsets(
  const TensorList<GPUBackend> &boxes_input, const cudaStream_t &stream) {
  vector<int> offsets {0};
  for (int i = 0; i < boxes_input.shape().size(); i++)
    offsets.push_back(boxes_input.shape().tensor_shape_span(i)[0] + offsets.back());

  auto offsets_data = boxes_offsets_.mutable_data<int>();
  MemCopy(offsets_data, offsets.data(), (curr_batch_size_ + 1) * sizeof(int), stream);

  return offsets_data;
}

void BoxEncoder<GPUBackend>::RunImpl(Workspace<GPUBackend> &ws) {
  const auto &boxes_input = ws.Input<GPUBackend>(kBoxesInId);
  const auto &labels_input = ws.Input<GPUBackend>(kLabelsInId);
  assert(ws.GetInputBatchSize(kBoxesInId) == ws.GetInputBatchSize(kLabelsInId));
  auto curr_batch_size = ws.GetInputBatchSize(kBoxesInId);

  const auto anchors_data = reinterpret_cast<const float4 *>(anchors_.data<float>());
  const auto anchors_as_cwh_data =
    reinterpret_cast<const float4 *>(anchors_as_center_wh_.data<float>());
  const auto boxes_data = reinterpret_cast<const float4 *>(boxes_input.data<float>());
  const auto labels_data = labels_input.data<int>();

  const auto buffers = ClearBuffers(ws.stream());

  auto boxes_offsets_data = CalculateBoxesOffsets(boxes_input, ws.stream());
  auto dims = CalculateDims(boxes_input);

  auto &boxes_output = ws.Output<GPUBackend>(kBoxesOutId);
  boxes_output.set_type(boxes_input.type());
  boxes_output.Resize(dims.first);
  auto boxes_out_data = reinterpret_cast<float4 *>(boxes_output.mutable_data<float>());

  auto &labels_output = ws.Output<GPUBackend>(kLabelsOutId);
  labels_output.set_type(labels_input.type());
  labels_output.Resize(dims.second);
  auto labels_out_data = labels_output.mutable_data<int>();

  const auto means_data = means_.data<float>();
  const auto stds_data = stds_.data<float>();

  if (!offset_)
    WriteAnchorsToOutput(boxes_out_data, labels_out_data, ws.stream());
  else
    ClearOutput(boxes_out_data, labels_out_data, ws.stream());

  Encode<BlockSize><<<curr_batch_size, BlockSize, 0, ws.stream()>>>(
    boxes_data,
    labels_data,
    boxes_offsets_data,
    anchors_count_,
    anchors_data,
    criteria_,
    boxes_out_data,
    labels_out_data,
    buffers.first,
    buffers.second,
    offset_,
    means_data,
    stds_data,
    scale_,
    anchors_as_cwh_data);
}

DALI_REGISTER_OPERATOR(BoxEncoder, BoxEncoder<GPUBackend>, GPU);

}  // namespace dali
