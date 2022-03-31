doc(title=("Operations", "="),
    options=":maxdepth: 2",
    entries=[
        # Must use explicit example_entry, for the doc mechanism to not expect README.py file
        # with index. Only .rst and .ipynb are detected as examples automatically
        example_entry("video_superres/README"),
        example_entry("pytorch/resnet50/pytorch-resnet50"),
        "pytorch/single_stage_detector/pytorch_ssd.rst",
        example_entry("tensorflow/resnet-n/README"),
        "tensorflow/yolov4/readme.rst",
        "tensorflow/efficientdet/README.rst",
        "paddle/index",
        "mxnet/mxnet-resnet50.ipynb",
        "detection_pipeline.ipynb",
        "webdataset-externalsource.ipynb",
    ])
