import pyndll as ndll
from timeit import default_timer as timer

# Initialize the allocators for ndll to use
ndll.Init(ndll.OpSpec("PinnedCPUAllocator", "Prefetch"),
          ndll.OpSpec("GPUAllocator", "Prefetch"))

iters = 100
batch_size = 32
num_threads = 1
cuda_stream = 0
device_id = 0

# Create our pipeline
pipe = ndll.Pipeline(batch_size,
                     num_threads,
                     cuda_stream,
                     device_id,
                     True, 0)

image_folder = "./benchmark_images"
img_type = ndll.RGB
interp_type = ndll.INTERP_LINEAR
output_type = ndll.FLOAT16
fast_resize = True

# Add a basic data reader
pipe.AddDataReader(
    ndll.OpSpec("BatchDataReader", "Prefetch")
    .AddArg("jpeg_folder", image_folder))

pipe.AddDecoder(
    ndll.OpSpec("TJPGDecoder", "Prefetch")
    .AddArg("output_type", img_type))

# Add a resize+crop+mirror op
if fast_resize:
    pipe.AddTransform(
        ndll.OpSpec("FastResizeCropMirrorOp", "Prefetch")
        .AddArg("random_resize", True)
        .AddArg("warp_resize", False)
        .AddArg("resize_a", 256)
        .AddArg("resize_b", 480)
        .AddArg("random_crop", True)
        .AddArg("crop_h", 224)
        .AddArg("crop_w", 224)
        .AddArg("mirror_prob", 0.5))
else:
    pipe.AddTransform(
        ndll.OpSpec("ResizeCropMirrorOp", "Prefetch")
        .AddArg("random_resize", True)
        .AddArg("warp_resize", False)
        .AddArg("resize_a", 256)
        .AddArg("resize_b", 480)
        .AddArg("random_crop", True)
        .AddArg("crop_h", 224)
        .AddArg("crop_w", 224)
        .AddArg("mirror_prob", 0.5))

pipe.AddTransform(
      ndll.OpSpec("NormalizePermuteOp", "Forward")
      .AddArg("output_type", output_type)
      .AddArg("mean", [128., 128., 128.])
      .AddArg("std", [1., 1., 1.])
      .AddArg("height", 224)
      .AddArg("width", 224)
      .AddArg("channels", 3))

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
