import pyndll as ndll
from timeit import default_timer as timer

# Initialize the allocators for ndll to use
ndll.Init(ndll.OpSpec("PinnedCPUAllocator"),
          ndll.OpSpec("GPUAllocator"))

iters = 100
batch_size = 256
num_threads = 5
cuda_stream = 0
device_id = 0

# Create our pipeline
pipe = ndll.Pipeline(batch_size,
                     num_threads,
                     cuda_stream,
                     device_id,
                     True, 0)

image_folder = "/home/tgale/data/alexnet-256"
img_type = ndll.RGB
interp_type = ndll.INTERP_LINEAR
output_type = ndll.FLOAT16

# Add a basic data reader
pipe.AddDataReader(
    ndll.OpSpec("BatchDataReader")
    .AddArg("stage", "Prefetch")
    .AddArg("jpeg_folder", image_folder))

# Add a hybrid jpeg decoder
pipe.AddDecoder(
    ndll.OpSpec("HuffmanDecoder")
    .AddArg("stage", "Prefetch")
    .AddExtraOutput("jpeg_meta"))

pipe.AddTransform(
    ndll.OpSpec("DCTQuantInvOp")
    .AddArg("stage", "Forward")
    .AddExtraInput("jpeg_meta")
    .AddArg("output_type", img_type))

# Add a bached crop+mirror+normalize+permute op
pipe.AddTransform(
    ndll.OpSpec("CropMirrorNormalizePermuteOp")
    .AddArg("stage", "Forward")
    .AddArg("output_type", output_type)
    .AddArg("random_crop", True)
    .AddArg("crop_h", 227)
    .AddArg("crop_w", 227)
    .AddArg("mirror_prob", 0.5)
    .AddArg("image_type", img_type)
    .AddArg("mean", [128., 128., 128.])
    .AddArg("std", [1., 1., 1.]))

# Setup the pipeline for execution
pipe.Build()

# Run once to allocator memory
pipe.RunPrefetch()
pipe.RunCopy()
pipe.RunForward()

# Run the pipeline
start_time = timer()
for i in range(iters):
    pipe.RunPrefetch()
    pipe.RunCopy()
    pipe.RunForward()
total_time = timer() - start_time
print("Images/second: %f" % ((batch_size * iters) / total_time))
