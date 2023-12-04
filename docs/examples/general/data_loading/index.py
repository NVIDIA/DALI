doc(
    title="Data Loading",
    underline_char="=",
    entries=[
        doc_entry(
            "external_input.ipynb",
            op_reference("fn.external_source", "Intro tutorial for external source"),
        ),
        doc_entry(
            "parallel_external_source.ipynb",
            op_reference("fn.external_source", "How to use parallel mode for external source"),
        ),
        doc_entry(
            "parallel_external_source_fork.ipynb",
            op_reference(
                "fn.external_source", "How to use parallel mode for external source in fork mode"
            ),
        ),
        doc_entry(
            "dataloading_lmdb.ipynb",
            [
                op_reference(
                    "fn.readers.caffe", "Example of reading data stored in LMDB in the Caffe format"
                ),
                op_reference(
                    "fn.readers.caffe2",
                    "Example of reading data stored in LMDB in the Caffe 2 format",
                ),
            ],
        ),
        doc_entry(
            "dataloading_recordio.ipynb",
            op_reference(
                "fn.readers.mxnet", "Example of reading data stored in the MXNet RecordIO format"
            ),
        ),
        doc_entry(
            "dataloading_tfrecord.ipynb",
            op_reference(
                "fn.readers.tfrecord",
                "Example of reading data stored in the TensorFlow TFRecord format",
            ),
        ),
        doc_entry(
            "dataloading_webdataset.ipynb",
            op_reference(
                "fn.readers.webdataset", "Example of reading data stored in the Webdataset format"
            ),
        ),
        doc_entry(
            "coco_reader.ipynb",
            op_reference("fn.readers.coco", "Example of reading a subset of COCO dataset"),
        ),
        doc_entry(
            "numpy_reader.ipynb",
            op_reference(
                "fn.readers.numpy",
                "Example of reading NumPy array files, "
                "including reading directly to GPU memory utilizing the GPUDirect storage",
            ),
        ),
    ],
)
