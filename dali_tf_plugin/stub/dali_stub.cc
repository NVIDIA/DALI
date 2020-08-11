// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/c_api.h"

void daliInitialize() {
}


void daliCreatePipeline(daliPipelineHandle *pipe_handle,
                        const char *serialized_pipeline,
                        int length,
                        int batch_size,
                        int num_threads,
                        int device_id,
                        int separated_execution,
                        int prefetch_queue_depth,
                        int cpu_prefetch_queue_depth,
                        int gpu_prefetch_queue_depth,
                        int enable_memory_stats) {
}


void daliDeserializeDefault(daliPipelineHandle *pipe_handle, const char *serialized_pipeline,
                            int length) {
}


void daliPrefetchUniform(daliPipelineHandle *pipe_handle, int queue_depth) {
}


void daliPrefetchSeparate(daliPipelineHandle *pipe_handle,
                          int cpu_queue_depth, int gpu_queue_depth) {
}


void daliSetExternalInput(daliPipelineHandle *pipe_handle, const char *name, device_type_t device,
                          const void *data_ptr, dali_data_type_t data_type, const int64_t *shapes,
                          int sample_dim, const char *layout_str, unsigned int flags) {
}


void daliSetExternalInputAsync(daliPipelineHandle *pipe_handle, const char *name,
                               device_type_t device, const void *data_ptr,
                               dali_data_type_t data_type, const int64_t *shapes,
                               int sample_dim, const char *layout_str, cudaStream_t stream,
                               unsigned int flags) {
}


void daliSetExternalInputTensors(daliPipelineHandle *pipe_handle, const char *name,
                                 device_type_t device, const void *const *data_ptr,
                                 dali_data_type_t data_type, const int64_t *shapes,
                                 int64_t sample_dim, const char *layout_str, unsigned int flags) {
}


void daliSetExternalInputTensorsAsync(daliPipelineHandle *pipe_handle, const char *name,
                                      device_type_t device, const void *const *data_ptr,
                                      dali_data_type_t data_type, const int64_t *shapes,
                                      int64_t sample_dim, const char *layout_str,
                                      cudaStream_t stream, unsigned int flags) {
}


void daliRun(daliPipelineHandle *pipe_handle) {
}


void daliOutput(daliPipelineHandle *pipe_handle) {
}


void daliShareOutput(daliPipelineHandle *pipe_handle) {
}


void daliOutputRelease(daliPipelineHandle *pipe_handle) {
}


int64_t* daliShapeAtSample(daliPipelineHandle* pipe_handle, int n, int k) {
}

int64_t* daliShapeAt(daliPipelineHandle* pipe_handle, int n) {
}

dali_data_type_t daliTypeAt(daliPipelineHandle* pipe_handle, int n) {
}

size_t daliNumTensors(daliPipelineHandle* pipe_handle, int n) {
}

size_t daliNumElements(daliPipelineHandle* pipe_handle, int n) {
}

size_t daliTensorSize(daliPipelineHandle* pipe_handle, int n) {
}

size_t daliMaxDimTensors(daliPipelineHandle* pipe_handle, int n) {
}

unsigned daliGetNumOutput(daliPipelineHandle* pipe_handle) {
}

device_type_t daliGetOutputDevice(daliPipelineHandle *pipe_handle, int id) {
}

void daliOutputCopy(daliPipelineHandle *pipe_handle, void *dst, int output_idx,
                    device_type_t dst_type, cudaStream_t stream, unsigned int flags) {
}

void daliOutputCopySamples(daliPipelineHandle *pipe_handle, void **dsts, int output_idx,
                           device_type_t dst_type, cudaStream_t stream, unsigned int flags) {
}


void daliCopyTensorNTo(daliPipelineHandle *pipe_handle, void *dst, int output_id,
                    device_type_t dst_type, cudaStream_t stream, int non_blocking) {
}

void daliCopyTensorListNTo(daliPipelineHandle *pipe_handle, void *dst, int output_id,
                           device_type_t dst_type, cudaStream_t stream, int non_blocking) {
}

void daliDeletePipeline(daliPipelineHandle* pipe_handle) {
}

void daliLoadLibrary(const char* lib_path) {
}

void daliGetReaderMetadata(daliPipelineHandle* pipe_handle, const char *reader_name,
                           daliReaderMetadata* meta) {
}

void daliGetExecutorMetadata(daliPipelineHandle* pipe_handle, daliExecutorMetadata **operator_meta,
                             size_t *operator_meta_num) {
}

void daliFreeExecutorMetadata(daliExecutorMetadata *operator_meta, size_t operator_meta_num) {
}
