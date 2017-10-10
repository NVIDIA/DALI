#include "ndll/pipeline/pipeline.h"

#include <algorithm>
#include <functional>
#include <memory>

namespace ndll {

void Pipeline::Build() {
  NDLL_ENFORCE(!built_, "\"Build()\" can only be called once");
  
  // Make sure the decoder is the first op in the pipeline
  if (decode_location_ == DECODE_FORWARD) {
    NDLL_ENFORCE(prefetch_ops_.size() == 0,
        "The Decoder is set to occur in the forward "
        "pipeline stage but prefetch operators exist");
  }
  NDLL_ENFORCE(prefetch_ops_.size() + forward_ops_.size() > 0,
      "Pipeline must have at least one operator");
  NDLL_ENFORCE(data_reader_ != nullptr,
      "The Pipeline must have a data reader to be built");

  // If we don't have a Parser set, add a default parser to the pipeline
  if (data_parser_ == nullptr) {
    data_parser_.reset(new DefaultParser);
  }

  // Note: In the case where we have no operators in the prefetch stage,
  // we need a way to get the output of the data reader + parser into a
  // Batch so that it can be copied to the gpu and passed to the forward
  // stage operators. For now, we insert a CopyOp to handle this
  if (prefetch_ops_.size() == 0) {
    OpPtr<CPUBackend> tmp(new CopyOp<CPUBackend>);
    prefetch_ops_.push_back(std::move(tmp));
  }

  // Read a single sample from the database and
  // parse it to get the input type
  data_reader_->Read(&input_datum_[0]);
  data_parser_->Run(input_datum_[0], &parsed_datum_[0], 0, 0);
  TypeMeta read_output_type = input_datum_[0].type();
  TypeMeta parsed_output_type = parsed_datum_[0].type();
  NDLL_ENFORCE(read_output_type.id() != NO_TYPE);
  NDLL_ENFORCE(parsed_output_type.id() != NO_TYPE);
  data_reader_->Reset();
  TypeMeta input_type = parsed_output_type;
  
  // Create buffers for intermediate pipeline results. We need 1
  // buffer for each cpu side op output, and 1 buffer for each
  // gpu side op input.
  //
  // For the Prefetch ops, we also need to set the output buffer
  // types so that the memory can be allocated prior to wrapping
  // individual samples in 'Datum' objects. The forward ops get
  // the whole batch at once, but we call these methods here as
  // well to avoid as much work as possible during training
  for (size_t i = 0; i < prefetch_ops_.size(); ++i) {
    BatchPtr<CPUBackend> tmp_cpu(new Batch<CPUBackend>);
    cpu_buffers_.push_back(std::move(tmp_cpu));
    prefetch_ops_[i]->SetOutputType(cpu_buffers_[i].get(), input_type);
    input_type = cpu_buffers_[i]->type();
  }

  // For the forward stage, we create a maximum of two buffers
  // and ping-pong back and forth between them. We always need
  // to create one to copy into regardless of how many ops are
  // to be executed in the forward stage
  for (size_t i = 0; i < std::min(forward_ops_.size()+1, 2lu); ++i) {
    BatchPtr<GPUBackend> tmp_gpu(new Batch<GPUBackend>);
    gpu_storage_.push_back(std::move(tmp_gpu));

    // We'll manage these buffers in bytes for simplicity
    gpu_storage_.back()->template data<uint8>();
  }

  // For all of the ops we will create batches, but share the
  // underlying data with one of the two 'gpu_storage' batches.
  // All forward ops are run in the same stream, so we can
  // guarantee that no op will corrupt the input/output for any
  // other op.
  {
    // We always need one buffer to copy into
    BatchPtr<GPUBackend> tmp_gpu(new Batch<GPUBackend>);
    gpu_buffers_.push_back(std::move(tmp_gpu));
    gpu_buffers_[0]->set_type(input_type);
  }
  
  for (size_t i = 0; i < forward_ops_.size(); ++i) {
    BatchPtr<GPUBackend> tmp_gpu(new Batch<GPUBackend>);
    gpu_buffers_.push_back(std::move(tmp_gpu));
    forward_ops_[i]->SetOutputType(gpu_buffers_[i+1].get(), input_type);
    input_type = gpu_buffers_[i+1]->type();
  }

  // Vector of shapes for all of the intermediate operator results as
  // well as the output gpu buffer
  intermediate_shapes_.resize(cpu_buffers_.size() + gpu_buffers_.size());

  // Size all the intermediate shapes for threads to write into
  for (auto &shape : intermediate_shapes_) {
    shape.resize(batch_size_);
  }

  // Set important meta-data for all the operators in the pipeline
  for (auto &op : prefetch_ops_) {
    op->set_num_threads(thread_pool_.size());
    op->set_batch_size(batch_size_);
    op->set_stream(stream_);
  }
  for (auto &op : forward_ops_) {
    op->set_num_threads(thread_pool_.size());
    op->set_batch_size(batch_size_);
    op->set_stream(stream_);
  }

  // TODO(tgale): Move these resize amounts to be based on hints from the
  // user. We could also have a setting where we run the pipeline and add
  // a small threshold.
  
  // TODO(tgale): The large number of memory allocations in the pipeline
  // interfere with training alot due to implicit synchronization w/
  // pinned memory allocations. For now, we presize the buffers to reasonable
  // sizes for their task. However, these sizes are basically fit to imagenet,
  // and to a jpeg quality of about 85. We should move to wrapping our backend
  // objects withing caching allocators.
  // for (auto &datum : input_datum_) {
  //   datum.set_type(read_output_type);
  //   datum.Resize({300000}); // 3000KB
  // }
  // for (auto &datum : parsed_datum_) {
  //   datum.set_type(parsed_output_type);
  //   datum.Resize({300000}); // 300KB
  // }

  // mega_buffer_.Resize({300000}); // 300KB
  // mega_buffer_gpu_.Resize({300000}); // 300KB
  
  // vector<Dims> tmp(1);
  // tmp[0].push_back(1500000 * batch_size_); // 1.5MB / sample
  // for (auto &buf : cpu_buffers_) buf->Resize(tmp);
  // for (auto &buf : gpu_buffers_) buf->Resize(tmp);
  
  // Mark the pipeline as built so we know it is safe to run
  built_ = true;
}

void Pipeline::RunPrefetch() {
  NDLL_ENFORCE(built_, "\"Build()\" must be called before the pipeline is executed");
  
  {
    TimeRange _tr("size-inference-loop");
    for (Index i = 0; i < batch_size_; ++i) {
      // Get a Datum from the reader
      data_reader_->Read(&input_datum_[i]);
      
      // Run type inference for this image on the whole pipeline
      thread_pool_.DoWorkWithID(std::bind(
              [this] (int data_idx, int tid) {
                // Run the Parsers
                data_parser_->Run(input_datum_[data_idx],
                    &parsed_datum_[data_idx], data_idx, tid);

                // Get the output shape for the cpu-side results
                intermediate_shapes_[0][data_idx] =
                  prefetch_ops_[0]->InferOutputShape(parsed_datum_[data_idx], data_idx, tid);
                
                Datum<CPUBackend> datum(intermediate_shapes_[0][data_idx]);
                for (size_t j = 1; j < prefetch_ops_.size(); ++j) {
                  intermediate_shapes_[j][data_idx] =
                    prefetch_ops_[j]->InferOutputShape(datum, data_idx, tid);
                  datum.Resize(intermediate_shapes_[j][data_idx]);
                }
                
                // Save the shape of the gpu-side copy buffer
                int offset = prefetch_ops_.size();
                intermediate_shapes_[offset][data_idx] =
                  intermediate_shapes_[offset - 1][data_idx];

                // Get the output shape for the gpu-side results
                ++offset;
                Datum<GPUBackend> datum_gpu(datum.shape());
                for (size_t j = 0; j < forward_ops_.size(); ++j) {
                  intermediate_shapes_[j + offset][data_idx] =
                    forward_ops_[j]->InferOutputShape(datum_gpu, data_idx, tid);
                  datum_gpu.Resize(intermediate_shapes_[j + offset][data_idx]);
                }
              }, i, std::placeholders::_1));
    }
    thread_pool_.WaitForWork();
  }
  
  {
    TimeRange _tr("resize-buffer-setup-prefetch");

    // Resize intermediate buffers and manage data sharing
    // on device for the forward stage of computation
    IntermediateBufferResizeAndSetup();

    // Setup the mega-buffer & distribute sub-buffers to
    // the ops in the forward pass
    MegaBufferSetupAndDistribution();
  }
  
  // Execute all the prefetch ops
  {
    TimeRange _tr("prefetch-work-loop");
    for (Index i = 0; i < batch_size_; ++i) {
      thread_pool_.DoWorkWithID(std::bind(
              [this] (int data_idx, int tid) {
                // We're going to ping-pong back and forth between these Datums
                // So we can cut the number of calls to "WrapSample()" in half.
                vector<Datum<CPUBackend>> datums(2);
                datums[1].WrapSample(cpu_buffers_[0].get(), data_idx);
                
                prefetch_ops_[0]->Run(parsed_datum_[data_idx], &datums[1], data_idx, tid);
                for (size_t j = 1; j < prefetch_ops_.size(); ++j) {
                  // Get the other datum to output this ops result into
                  datums[!(j&1)].WrapSample(cpu_buffers_[j].get(), data_idx);
                  prefetch_ops_[j]->Run(datums[j&1], &datums[!(j&1)], data_idx, tid);
                }

                // Give the forward ops a chance to set up
                // parameters for their kernel launches
                for (size_t j = 0; j < forward_ops_.size(); ++j) {
                  forward_ops_[j]->BatchedParameterSetupPerDatum(
                      *gpu_buffers_[j], gpu_buffers_[j+1].get(), data_idx, tid);
                }
              }, i, std::placeholders::_1));
    }
    thread_pool_.WaitForWork();
  }
}

void Pipeline::RunCopy() {
  Batch<CPUBackend> &src = *cpu_buffers_[cpu_buffers_.size()-1];
  Batch<GPUBackend> *dst = gpu_buffers_[0].get();

  // Copy the data to the GPU in the main stream
  CUDA_CALL(cudaMemcpyAsync(
          dst->raw_data(),
          src.raw_data(),
          src.nbytes(),
          cudaMemcpyHostToDevice,
          stream_));

  // Copy the mega-buffer to GPU in the main stream
  CUDA_CALL(cudaMemcpyAsync(
          mega_buffer_gpu_.raw_data(),
          mega_buffer_.raw_data(),
          mega_buffer_.nbytes(),
          cudaMemcpyHostToDevice,
          stream_));
}

void Pipeline::RunForward() {
  TimeRange _tr("RunForward");
  NDLL_ENFORCE(built_,
      "\"Build()\" must be called before the pipeline is executed");

  // TODO(tgale): figure out a cleaner way to do this. We need
  // to set the stream here each time so that frameworks like
  // C2 that have different threads running through this method
  // on any given iteration have the correct stream to maintain
  // the dependency between the copy and these kernels. Each op
  // should probably deal with this independently, allthough
  // this could interfere with any use of npp external to this
  // code.
  nppSetStream(stream_);
  
  // Run all the forward ops
  for (size_t i = 0; i < forward_ops_.size(); ++i) {
    forward_ops_[i]->Run(*gpu_buffers_[i], gpu_buffers_[i+1].get());
  }
}

void Pipeline::IntermediateBufferResizeAndSetup() {
  // Resize the host-side intermediate buffers
  for (size_t i = 0; i < cpu_buffers_.size(); ++i) {
    cpu_buffers_[i]->Resize(intermediate_shapes_[i]);
  }


  // Calculate the maximum number of bytes needed for each
  // of the gpu storage buffers for the forward pass. Each
  // buffer will be used for every other buffer in the
  // forward pass, so we need to find how much memory will
  // be needed for even numbered buffers and odd numbered
  // buffers separately
  vector<size_t> max_bytes(2, 0);
  for (size_t i = 0; i < gpu_buffers_.size(); ++i) {
    Index required_size = 0;
    for (auto &dims : intermediate_shapes_[i + cpu_buffers_.size()]) {
      Index tmp = 1;
      for (auto &val : dims) {
        tmp *= val;
      }
      required_size += tmp;
    }

    size_t required_bytes = required_size * gpu_buffers_[i]->type().size();
    max_bytes[i&1] = std::max(max_bytes[i&1], required_bytes);
  }

  // Resize the two storage buffers
  for (size_t i = 0; i < std::min(forward_ops_.size()+1, 2lu); ++i) {
    gpu_storage_[i]->Resize({{(Index)max_bytes[i]}});
  }

  // Share the gpu data with the other gpu buffers and
  // resize them to their required size for this iteration
  for (size_t i = 0; i < gpu_buffers_.size(); ++i) {
    gpu_buffers_[i]->ShareData(*gpu_storage_[i&1]);

    // Note: We've already allocated the maximum amount of data
    // that these buffers need, so this resize should not trigger
    // any memory allocations
    gpu_buffers_[i]->Resize(
        intermediate_shapes_[i + cpu_buffers_.size()]);
    NDLL_ENFORCE(gpu_buffers_[i]->shares_data(),
        "GPU buffers should still be sharing data, something went wrong.");
  }
}

void Pipeline::MegaBufferSetupAndDistribution() {
  // Query all the forward ops for mega-buffer sizes
  size_t total_bytes = 0;
  vector<size_t> offsets;
  vector<int> num_buff_for_op(forward_ops_.size(), 0);
  for (size_t i = 0; i < forward_ops_.size(); ++i) {
    const vector<size_t>& sizes =
      forward_ops_[i]->GetBatchedParameterSize();
    num_buff_for_op[i] = sizes.size();
    for (auto &num_bytes : sizes) {
      // Align the start of each buffer to 8-bytes
      size_t aligned_num_bytes = round_up_to_8(num_bytes);
      offsets.push_back(total_bytes);
      total_bytes += aligned_num_bytes;
    }
  }
  offsets.push_back(total_bytes);
  
  mega_buffer_.Resize({(Index)total_bytes});
  mega_buffer_gpu_.Resize({(Index)total_bytes});

  // Hand out SubTensors for all the ops batched parameters
  int buffer_id = 0;
  for (size_t i = 0; i < forward_ops_.size(); ++i) {
    vector<SubTensor<CPUBackend>> cpu_buffers(num_buff_for_op[i]);
    vector<SubTensor<GPUBackend>> gpu_buffers(num_buff_for_op[i]);
    for (int j = 0; j < num_buff_for_op[i]; ++j) {
      SubTensor<CPUBackend> sub_buffer(&mega_buffer_,
          offsets[buffer_id], offsets[buffer_id+1]);
      SubTensor<GPUBackend> sub_buffer_gpu(&mega_buffer_gpu_,
          offsets[buffer_id], offsets[buffer_id+1]);
      cpu_buffers[j] = sub_buffer;
      gpu_buffers[j] = sub_buffer_gpu;
      ++buffer_id;
    }
    forward_ops_[i]->
      SetBatchedParameterBuffers(cpu_buffers, gpu_buffers);
    forward_ops_[i]->
      BatchedParameterSetup(*gpu_buffers_[i], gpu_buffers_[i+1].get());
  }
}

} // namespace ndll
