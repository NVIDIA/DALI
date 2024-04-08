Q&A
***

Q: How do I know if DALI can help me?
#####################################
A: You need to check our docs first and see if DALI operators cover your use case. Then, try to run
a couple of iterations of your training with a fixed data source - generating the batch once and
reusing it over the test run to see if you can train faster without any data processing. If so,
then the data processing is a bottleneck, and in that case, DALI may help. This topic is covered
in detail in
`the GTC'22 talk <https://www.nvidia.com/gtc/session-catalog/#/session/1636559250287001p4DG>`__.

Q: What data formats does DALI support?
#######################################
A: DALI can load most of the image file formats (JPEG, PNG, TIFF, JPEG2000 and more) as well
as audio (WAV, OGG, FLAC) and video (avi, mp4, mkv, mov, H264, HEVC, MPEG4, MJPEG, VP8, VP9).
The files can be stored individually or grouped in storage containers from common DL frameworks -
TFRecord, RecordIO, caffe/caffe2 LMDB. WebDataset is also supported. In case there's no native
support for a particular format, the data can be loaded and parsed in user-provided python code
and supplied via the external source operator.

Q: How does DALI differ from TF, PyTorch, MXNet, or other FWs
#############################################################
A: The main difference is that the data preprocessing, and augmentations are GPU accelerated,
and the processing is done for the whole batch at the time to improve GPU utilization. Also,
it can be used in multiple different FW - TF, MXNet, PyTorch, PaddlePaddle - so you are sure
that the data processing is bit-exact, which is important when you move models between FWs
and see some discrepancies in the accuracy. Moreover, it provides a convenient way to switch
data processing between GPU and CPU to utilize all available computational resources.

Q: What to do if DALI doesn't cover my use case?
################################################
A: You can experiment first with writing a custom operator in Python to check if you can create
a data pipeline you like (you can even use CuPy inside to utilize GPU). Then you can create
a plugin with an operator running the native code or use Numba to convert python code to
a native operator. You can also become a contributor and make your work a part of
the DALI code base - we would be more than happy to review and accept any PR with new
functionality.

Q: How to use DALI for inference?
#################################
A: You can easily employ DALI for inference together with the `Triton Inference Server <https://developer.nvidia.com/nvidia-triton-inference-server>`__.
We developed a dedicated `DALI Backend <https://github.com/triton-inference-server/dali_backend>`__
so all you need to do is to provide a description of the processing pipeline to the Triton, and add
DALI to the model ensemble. For more information about using DALI with Triton, please refer to the
`DALI Backend documentation <https://github.com/triton-inference-server/dali_backend#how-to-use>`__

Q: How big is the speedup of using DALI compared to loading using OpenCV? Especially for JPEG images.
######################################################################################################
A: DALI utilizes nvJPEG to accelerate JPEG decoding. It achieves up to 2.5x speedup with
NVIDIA A100 Ampere GPU - `see for details <https://developer.nvidia.com/blog/loading-data-fast-with-dali-and-new-jpeg-decoder-in-a100/>`__.
In case of other image formats, for which there's no GPU accelerated decoding, DALI uses either OpenCV
(like for PNG) or dedicated library directly, like libtiff. In this case, the performance should
be comparable.

Q: Can you use DALI with DeepStream?
####################################
A: There is no reason why models trained with DALI cannot be used for inference with DeepStream.
DeepStream mostly focuses on online video and audio inference pipelines, DALI can do this only
for offline scenarios. There is no direct integration between these two solutions.

Q: How to control the number of frames in a video reader in DALI?
#################################################################
A: It can be controlled by the `sequence_length` argument.

Q: Can DALI volumetric data processing work with ultrasound scans?
##################################################################
A: DALI doesn't support any domain-specific data formats like `NIfTI <https://nifti.nimh.nih.gov/>`__,
but in most cases, the data is first processed offline and converted to a more
data-processing-friendly format like NumPy arrays (including initial normalization and conversion).
If the data is available in DALI-supported format there is no reason why it cannot be processed
no matter if it is an ultrasound or CT scan.

Please be advised, that DALI does not support 3D reconstruction, either from sequences of 2D
ultrasound scans or CT sinograms.

Q: How to debug a DALI pipeline?
################################
A: Just recently DALI added `an eager debug mode <examples/general/debug_mode.html>`__ so
the output of each operator can be instantly evaluated and Python code can be added inside
the pipeline for the prototyping purpose.

Q: Can I access the contents of intermediate data nodes in the pipeline?
########################################################################
A: In the pipeline mode it is not possible, however, thanks to the recently introduced
`debug mode <examples/general/debug_mode.html>`__ it can be done. For performance
reasons, this feature is intended only for debugging and prototyping.

Q: When will DALI support the XYZ operator?
###########################################
A: We cannot commit to any timeline to add any particular operator. Still, we are open to external
contributions. On top of that, every user can extend DALI on his own without modifying its code
base, please check `this page <examples/custom_operations/index.html>`__ for more details.

Q: How should I know if I should use a CPU or GPU operator variant?
###################################################################
A: It depends on the particular use case. I would start by checking GPU  utilization to see if
throwing more work on the GPU won't slow things down, or just place everything on the CPU first
and then gradually move things to the GPU and see if that helps. You can check
`our GTC22 talk <https://www.nvidia.com/gtc/session-catalog/#/session/1636559250287001p4DG>`__
or `the blog post <https://developer.nvidia.com/blog/case-study-resnet50-dali/>`__ to see how
to do it step by step.

Q: How can I provide a custom data source/reading pattern to DALI?
##################################################################
A: You can define any custom data loading pattern using python and
`external source operator <examples/general/data_loading/external_input.html>`__. To make it
faster please use `its parallel capability <examples/general/data_loading/parallel_external_source.html>`__.

Q: Does DALI have any profiling capabilities?
#############################################
A: DALI doesn't have any built-in profiling capabilities, still it utilizes NVTX ranges
and has a dedicated domain (so it is easy to find in the profile) to show its operations. So you can
capture the profile using `NVIDIA Nsight Systems <https://developer.nvidia.com/nsight-systems>`__
or any Deep Learning profile that also supports NVTX markers.

Q: Does DALI support multi GPU/node training?
#############################################
A: Yes, DALI supports data-parallel and distributed data-parallel strategies (you can read more
about these strategies `here <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#comparison-between-dataparallel-and-distributeddataparallel>`__).
Its shards data into non-overlapping pieces using the number of shards (world size) and shard id (global rank), and
uses device id to identify the GPU used in the particular node (local rank).

More details can be also found it `this documentation section <advanced_topics_sharding.html>`__

Q: How to report an issue/RFE or get help with DALI usage?
##########################################################
A: DALI is an open-source project hosted on GitHub, you can ask questions and report issues
using `this link <https://github.com/NVIDIA/DALI/issues>`__ directly.

Q: Can DALI accelerate the loading of the data, not just processing?
####################################################################
A: DALI mostly focuses on processing acceleration, as in most cases the input data is compressed
(audio, video, or images) and the input data is relatively small compared to the raw decoded output.
Still, there are cases, where data is not compressed and loading it directly to the GPU is feasible.
To support that case DALI can use `GPUDirect Storage <https://developer.nvidia.com/gpudirect-storage>`__
inside Numpy GPU reader to bypass CPU and load the data directly to the GPU.

Q: How can I obtain DALI?
#######################################################
A: DALI is available as a prebuilt python wheel binary -
`see to learn how to install it <https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html>`__
or as `a source code <https://github.com/NVIDIA/DALI>`__ that can be built on your own.

Q: Which OS does DALI support?
##############################
A: DALI does support all major Linux distributions and indirectly Windows through
`WSL <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>`__. Regrettably, MacOS
is not supported.

Q: Where can I find the list of operations that DALI supports?
##############################################################
A: You can find a comprehensive list of operators available `here <supported_ops.html>`__.

Q: Can I send a request to the Triton server with a batch of samples of different shapes (like files with different lengths)?
#############################################################################################################################
A: Batch processing is one of main DALI paradigms. On the other hand, Triton Inference Server
supports a uniform batch by default. However, by enabling
a `ragged batching <https://github.com/triton-inference-server/server/blob/v2.26.0/docs/user_guide/ragged_batching.md>`__
you can send non-uniform batches and process them successfully.
`Here <https://github.com/triton-inference-server/dali_backend/blob/7d51c7299dd66964097f839501e18f3b579cc306/qa/L0_DALI_GPU_ensemble/client.py#L31>`__
you can find an example of using ragged batching feature with DALI Backend.

Q: I have heard about the new data processing framework XYZ, how is DALI better than it?
########################################################################################
A: DALI is a library that aims to GPU accelerate certain workloads we see that suffer the most
due to being CPU bottleneck. There are many cases not covered by DALI, or where DALI
can be suboptimal, and these are the places where other solutions could shine.

What is worth remembering, there is a lot of advertised optimizations in other libraries that
come with the cost of lower accuracy in the training or inference process - DALI has proved
itself in MLPerf benchmarks and `NVIDIA Deep Learning Examples <https://github.com/NVIDIA/DeepLearningExamples>`__
where not only speed but also accuracy matters. So the user is sure that DALI doesn't cut corners.

Q: Is DALI compatible with other GPUs?
######################################
A: When it comes to the question if DALI supports non-NVIDIA GPUs, the answer is no.
DALI GPU implementations are written in CUDA. However there are open source community efforts
that are enabling running CUDA-based applications on other GPU architectures, but DALI
doesn't officially support it.

Q: When to use DALI and when RAPIDS?
####################################
A: RAPIDS is better suited for general-purpose machine learning and data analytics.
DALI is a specialized tool for Deep Learning workflows, and it's aimed to accelerate dense data
(such as images, video, audio) processing and to overlap the preprocessing with
the network forward/backward passes.

Q: Is Triton + DALI still significantly better than preprocessing on CPU, when minimum latency i.e. batch_size=1 is desired?
############################################################################################################################
It depends on what base implementation we compare to, but generally, DALI gives
the most benefit to the throughput of the training/inference because of the batch processing
that can utilize massive parallelism of the GPUs. Still, the GPU implementations of DALI operators
are optimized and fast, so it might reduce the inference latency.

Q: Are there any examples of using DALI for volumetric data?
############################################################
A: Yes, e.g DALI was used to achieve high performance in NVIDIA’s MLPerf submission for UNet3D.
You can read an interesting article about it `here <https://developer.nvidia.com/blog/accelerating-medical-image-processing-with-dali>`__.
You can see the DALI pipeline that was used `in this example <https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Segmentation/nnUNet/data_loading/dali_loader.py>`__.

Q: Where can I find more details on using the image decoder and doing image processing?
#######################################################################################
A: You can always refer to `the relevant section of the DALI documentation <examples>`__
where you can find multiple examples of DALI used in different use-cases. For the start,
you can also watch `our introductory talk on this GTC <https://www.nvidia.com/gtc/session-catalog/#/session/1636566824182001pODM>`__.

Q: Does DALI utilize any special NVIDIA GPU functionalities?
############################################################
A: Yes, DALI uses `NVJPEG <https://developer.nvidia.com/blog/loading-data-fast-with-dali-and-new-jpeg-decoder-in-a100/>`__ -
special HW unit offloading JPEG image decoding, `NVDEC <https://developer.nvidia.com/nvidia-video-codec-sdk>`__ -
HW video decoder, `GPUDirect Storage <https://developer.nvidia.com/gpudirect-storage>`__ -
the ability to load data directly to the GPU to avoid a slow round trip through CPU.

Q: Can DALI operate without GPU?
################################
A: Yes. Vast majority of operators have CPU and GPU variants and a pipeline where all operators are
run on CPU doesn't require a GPU to run. However, DALI is predominantly a GPU library and CPU
operators are not as thoroughly optimized.
The main goal of this functionality is to enable the development of the DALI pipeline on
machines where GPU is not available (like laptops), with an ability to later deploy the DALI
pipeline on a GPU-capable cluster.

Q: Can I use DALI in the Triton server through a Python model?
##############################################################
A: You could do that if the Python used by the server has DALI installed but for
the best performance, we encourage you to use the dedicated DALI backend. It skips
the Python layer and optimizes the interaction between the Triton server and the DALI pipeline.

Q: Can the Triton model config be auto-generated for a DALI pipeline?
#####################################################################
A: Not yet but we are actively working on that feature and we expect to provide
model config auto-generation for the DALI Backend soon.

Q: How easy is it to integrate DALI with existing pipelines such as PyTorch Lightning?
#######################################################################################
A: It is very easy to integrate with PyTorch Lightning thanks to the PyTorch iterator.
There is a dedicated example available `here <examples/frameworks/pytorch/pytorch-lightning.html>`__.

Q: Does DALI typically result in slower throughput using a single GPU versus using multiple PyTorch worker threads in a data loader?
####################################################################################################################################
A: In the case of CPU execution, DALI also uses multiple worker threads.
Using DALI should produce a better performance in most cases, even for one GPU.
Of course, the details can depend on the particular CPU and GPU and the pipeline itself,
as well as the current GPU utilization before introducing DALI. You can check
`our GTC22 talk <https://www.nvidia.com/gtc/session-catalog/#/session/1636559250287001p4DG>`__
or `the blog post <https://developer.nvidia.com/blog/case-study-resnet50-dali>`__ to see this in practice.

Q: Will labels, for example, bounding boxes, be adapted automatically when transforming the image data? For example when rotating/cropping, etc. If so how?
###########################################################################################################################################################
A: The meta-data, like bounding boxes or coordinates, will not be adapted automatically with
the data but DALI has a set of operators, e.g.
`bbox_paste <operations/nvidia.dali.fn.bbox_paste.html>`__,
`random_bbox_crop <operations/nvidia.dali.fn.random_bbox_crop.html>`__ for bounding boxes or
`coord_transform <operations/nvidia.dali.fn.coord_transform.html>`__ for sets of coordinates.
You can find an example `here <examples/use_cases/detection_pipeline.html>`__.

Q: How easy is it, to implement custom processing steps? In the past, I had issues with calculating 3D Gaussian distributions on the CPU. Would this be possible using a custom DALI function?
################################################################################################################################################################################################
A: There are several ways to do it. You can write custom operators in C++/CUDA, or run arbitrary
Python code via the Python function and Numba operators. You can learn more about this topic
`here <examples/custom_operations/index.html>`__.

Q: Is DALI available in Jetson platforms such as the Xavier AGX or Orin?
########################################################################
A: At the moment we are not releasing binaries for Jetson, but it should be possible to build
DALI from source. You can learn more about the exact steps
`here <compilation.html#cross-compiling-for-aarch64-jetson-linux-docker>`__.

Q: Is it possible to get data directly from real-time camera streams to the DALI pipeline?
##########################################################################################
A: There is no dedicated way of dealing with camera streams in DALI but you can implement it using
`the fn.external_source operator <examples/general/data_loading/external_input.html>`__.
It allows you to use a Python function or an iterator to provide the data so if your camera stream
is accessible from Python - this is the way to go.

Q: What is the advantage of using DALI for the distributed data-parallel batch fetching, instead of the framework-native functions?
###################################################################################################################################
A: By using DALI you accelerate not only data-loading but also the whole preprocessing pipeline -
so you get the benefit of batch processing on the GPU and overlapping the preprocessing with
the training. DALI also has the prefetching queue which means that it can preprocess a few batches
ahead of time to maximize the throughput.