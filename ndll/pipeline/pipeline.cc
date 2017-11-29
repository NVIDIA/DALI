#include "ndll/pipeline/pipeline.h"

#include <algorithm>
#include <functional>
#include <memory>

namespace ndll {

// TODO(tgale): The constraint that GPU ops cannot produce CPU
// outputs is arbitrary. We could easily enable cpu/gpu outputs
// for gpu ops, do we want to do this?
void Pipeline::AddOperator(OpSpec spec) {
  NDLL_ENFORCE(!built_, "Alterations to the pipeline after "
      "\"Build()\" has been called are not allowed");

  // Validate op device
  string device = spec.GetArgument<string>("device", "cpu");
  NDLL_ENFORCE(device == "cpu" || device == "gpu", "Invalid "
      "device argument \"" + device + "\". Valid options are "
      "\"cpu\" or \"gpu\"");

  // Verify the inputs to the op
  for (int i = 0; i < spec.NumInput(); ++i) {
    string input_name = spec.InputName(i);
    string input_device = spec.InputDevice(i);
    auto it = edge_names_.find(input_name);

    NDLL_ENFORCE(it != edge_names_.end(), "Input '" + input_name +
        "' to op '" + spec.name() + "' is not known to the pipeline.");

    // Table of possible scenarios:
    // Op location / requested input type / data location
    // cpu / cpu / cpu -> everything is fine
    // cpu / cpu / gpu -> error, data does not exist on cpu
    // cpu / gpu / cpu -> error, cpu op not allowed to have gpu inputs
    // cpu / gpu / gpu -> both of above errors
    // gpu / cpu / cpu -> need to use contiguous version
    // gpu / cpu / gpu -> error, data not in specified location
    // gpu / gpu / cpu -> need to insert copy to device
    // gpu / gpu / gpu -> everything is fine
    string error_str = "(op: '" + spec.name() + "', input: '" +
      input_name + "')";
      
    if (device == "cpu") {
      NDLL_ENFORCE(input_device == "cpu", "cpu ops can only take cpu "
          "inputs. " + error_str);
      NDLL_ENFORCE(it->second.has_cpu, "cpu input requested by op exists "
          "only on gpu. " + error_str);
    } else if (input_device == "cpu") {
      NDLL_ENFORCE(it->second.has_cpu, "cpu input requested by op exists "
          "only on gpu. " + error_str);
      SetupCPUInput(it, i, &spec);
    } else {
      SetupGPUInput(it);
    }
  }

  // Verify and record the outputs of the op
  for (int i = 0; i < spec.NumOutput(); ++i) {
    string output_name = spec.OutputName(i);
    string output_device = spec.OutputDevice(i);
    string error_str = "(op: '" + spec.name() + "', output: '" +
      output_name + "')";
    
    auto it = edge_names_.find(output_name);
    NDLL_ENFORCE(it == edge_names_.end(), "Output name '" +
        output_name + "' conflicts with existing intermediate "
        "result name. " + error_str);

    // Validate output data conforms to graph constraints
    if (device == "cpu") {
      NDLL_ENFORCE(output_device == "cpu", "cpu ops can only produce "
          "cpu outputs." + error_str);
    } else {
      NDLL_ENFORCE(output_device == "gpu", "gpu ops can only produce "
          "gpu outputs." + error_str);
    }
    
    EdgeMeta meta = NewEdge(device);
    NDLL_ENFORCE(edge_names_.insert({output_name, meta}).second,
        "Output name insertion failure.");
  }

  // Add the operator to the graph
  PrepareOpSpec(&spec);
  graph_.AddOp(spec);
}

void Pipeline::Build() {
  /*
  NDLL_ENFORCE(!built_, "\"Build()\" can only be called once");
  
  // Make sure the decoder is the first op in the pipeline
  if (decode_location_ == DECODE_FORWARD) {
    NDLL_ENFORCE(prefetch_ops_.size() == 0,
        "The Decoder is set to occur in the forward "
        "pipeline stage but prefetch operators exist");
  }
  NDLL_ENFORCE(data_reader_ != nullptr,
      "The Pipeline must have a data reader to be built");

  // If we don't have a Parser set, add a default parser to the pipeline
  if (data_parser_ == nullptr) {
    data_parser_.reset(
        new DefaultParser(
            OpSpec("DefaultParser")
            .AddArg("batch_size", batch_size_)
            .AddArg("num_threads", num_threads())
            .AddArg("cuda_stream", (int64)stream_)
            )
        );
  }

  // Note: In the case where we have no operators in the prefetch stage,
  // we need a way to get the output of the data reader + parser into a
  // Batch so that it can be copied to the gpu and passed to the forward
  // stage operators. For now, we insert a CopyOp to handle this
  if (prefetch_ops_.size() == 0) {
    OpPtr<CPUBackend> tmp(
        new CopyOp<CPUBackend>(
            OpSpec("CopyOp")
            .AddArg("batch_size", batch_size_)
            .AddArg("num_threads", num_threads())
            .AddArg("cuda_stream", (int64)stream_)
            )
        );
    prefetch_ops_.push_back(std::move(tmp));
  }

  // Read a single sample from the database and
  // parse it to get the input type
  data_reader_->Read(&input_sample_[0]);
  data_parser_->Parse(input_sample_[0], &parsed_sample_[0], 0, 0);
  TypeInfo read_output_type = input_sample_[0].type();
  TypeInfo parsed_output_type = parsed_sample_[0].type();
  NDLL_ENFORCE(IsValidType(read_output_type));
  NDLL_ENFORCE(IsValidType(parsed_output_type));
  data_reader_->Reset();
  TypeInfo input_type = parsed_output_type;
  
  // Create buffers for intermediate pipeline results. We need 1
  // buffer for each cpu side op output, and 1 buffer for each
  // gpu side op input.
  //
  // For the Prefetch ops, we also need to set the output buffer
  // types so that the memory can be allocated prior to wrapping
  // individual samples in 'Sample' objects. The forward ops get
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
    gpu_storage_.back()->template mutable_data<uint8>();
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

  // NOTE: The large number of memory allocations in the pipeline
  // interfere with training alot due to implicit synchronization w/
  // pinned memory allocations. We use the user-given hint to pre-size
  // our buffers to avoid this slowdown while the buffer sizes stabilize.
  // We could use caching allocator wrappers, but this could potentially
  // interfere with the rest of the applications allocator and cause false
  // out of memory errors.
  
  for (auto &sample : input_sample_) {
    sample.set_type(read_output_type);
    sample.Resize({(Index)pixels_per_image_hint_});
  }
  for (auto &sample : parsed_sample_) {
    sample.set_type(parsed_output_type);
    sample.Resize({(Index)pixels_per_image_hint_});
  }

  // NOTE: We don't presize the mega-buffer, as the size of this usually
  // does not vary over time and has no relation to the size of the images.
  // If we wanted to do this, we could run over the forward ops and get
  // estimates of their memory requirements.

  // Resize the intermediate host & gpu storage
  vector<Dims> tmp(batch_size_, {(Index)pixels_per_image_hint_});
  for (auto &buf : cpu_buffers_) buf->Resize(tmp);
  for (auto &buf : gpu_storage_) buf->Resize(tmp);
  
  // mark the pipeline as built so we know it is safe to run
  built_ = true;
  */
}

void Pipeline::RunCPU() {
  /*
  NDLL_ENFORCE(built_, "\"Build()\" must be called before the pipeline is executed");
  
  {
    TimeRange _tr("size-inference-loop");
    for (Index i = 0; i < batch_size_; ++i) {
      // Get a Sample from the reader
      data_reader_->Read(&input_sample_[i]);
      
      // Run type inference for this image on the whole pipeline
      thread_pool_.DoWorkWithID(std::bind(
              [this] (int data_idx, int tid) {
                // Run the Parsers
                data_parser_->Parse(input_sample_[data_idx],
                    &parsed_sample_[data_idx], data_idx, tid);

                // Get the output shape for the cpu-side results
                intermediate_shapes_[0][data_idx] =
                  prefetch_ops_[0]->InferOutputShape(parsed_sample_[data_idx], data_idx, tid);
                
                Sample<CPUBackend> sample(intermediate_shapes_[0][data_idx]);
                for (size_t j = 1; j < prefetch_ops_.size(); ++j) {
                  intermediate_shapes_[j][data_idx] =
                    prefetch_ops_[j]->InferOutputShape(sample, data_idx, tid);
                  sample.Resize(intermediate_shapes_[j][data_idx]);
                }
                
                // Save the shape of the gpu-side copy buffer
                int offset = prefetch_ops_.size();
                intermediate_shapes_[offset][data_idx] =
                  intermediate_shapes_[offset - 1][data_idx];

                // Get the output shape for the gpu-side results
                ++offset;
                Sample<GPUBackend> sample_gpu(sample.shape());
                for (size_t j = 0; j < forward_ops_.size(); ++j) {
                  intermediate_shapes_[j + offset][data_idx] =
                    forward_ops_[j]->InferOutputShape(sample_gpu, data_idx, tid);
                  sample_gpu.Resize(intermediate_shapes_[j + offset][data_idx]);
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
                // We're going to ping-pong back and forth between these Samples
                // So we can cut the number of calls to "WrapSample()" in half.
                vector<Sample<CPUBackend>> samples(2);
                samples[1].WrapSample(cpu_buffers_[0].get(), data_idx);
                
                prefetch_ops_[0]->Run(parsed_sample_[data_idx], &samples[1], data_idx, tid);
                for (size_t j = 1; j < prefetch_ops_.size(); ++j) {
                  // Get the other sample to output this ops result into
                  samples[!(j&1)].WrapSample(cpu_buffers_[j].get(), data_idx);
                  prefetch_ops_[j]->Run(samples[j&1], &samples[!(j&1)], data_idx, tid);
                }

                // Give the forward ops a chance to set up
                // parameters for their kernel launches
                for (size_t j = 0; j < forward_ops_.size(); ++j) {
                  forward_ops_[j]->BatchedParameterSetupPerSample(
                      *gpu_buffers_[j], gpu_buffers_[j+1].get(), data_idx, tid);
                }
              }, i, std::placeholders::_1));
    }
    thread_pool_.WaitForWork();
  }
  */
}

void Pipeline::RunCopy() {
  /*
  Batch<CPUBackend> &src = *cpu_buffers_[cpu_buffers_.size()-1];
  Batch<GPUBackend> *dst = gpu_buffers_[0].get();

  // Copy the data to the GPU in the main stream
  CUDA_CALL(cudaMemcpyAsync(
          dst->raw_mutable_data(),
          src.raw_data(),
          src.nbytes(),
          cudaMemcpyHostToDevice,
          stream_));

  // Copy the mega-buffer to GPU in the main stream
  CUDA_CALL(cudaMemcpyAsync(
          mega_buffer_gpu_.raw_mutable_data(),
          mega_buffer_.raw_data(),
          mega_buffer_.nbytes(),
          cudaMemcpyHostToDevice,
          stream_));
  */
}

void Pipeline::RunGPU() {
  /*
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
  */
}

void Pipeline::SetupCPUInput(std::map<string, EdgeMeta>::iterator it,
    int input_idx, OpSpec *spec) {
  if (!it->second.has_contiguous) {
    OpSpec make_contiguous_spec =
      OpSpec("MakeContiguous")
      .AddArg("device", "internal")
      .AddInput(it->first, "cpu")
      .AddOutput("contiguous_" + it->first, "cpu");
    PrepareOpSpec(&make_contiguous_spec);
    graph_.AddOp(make_contiguous_spec);
  }

  // Update the OpSpec to use the contiguous input
  auto input_strs = spec->mutable_input(input_idx);
  NDLL_ENFORCE(input_strs->first == it->first, "Input at index " +
      std::to_string(input_idx) + " does not match input iterator "
      "name (" + input_strs->first + " v. " + it->first + ").");
  input_strs->first = "contiguous_" + input_strs->first;
}

void Pipeline::SetupGPUInput(std::map<string, EdgeMeta>::iterator it) {
  if (it->second.has_gpu) return;
  OpSpec copy_to_dev_spec =
    OpSpec("CopyToDevice")
    .AddArg("device", "internal")
    .AddInput(it->first, "cpu")
    .AddOutput(it->first, "gpu");
  PrepareOpSpec(&copy_to_dev_spec);
  graph_.AddOp(copy_to_dev_spec);
}

void Pipeline::MegaBufferSetupAndDistribution() {
  /*
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
  */
}

void Pipeline::PrepareOpSpec(OpSpec *spec) {
  // Add batch_size, num_threads, cuda stream, and
  // image size hint as arguments for the Op to
  // optionally leverage
  spec->AddArg("batch_size", batch_size_)
    .AddArg("num_threads", num_threads())
    .AddArg("cuda_stream", (int64)stream_)
    .AddArg("pixels_per_image_hint", pixels_per_image_hint_);
}

} // namespace ndll
