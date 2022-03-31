doc(title=("Data Loading", "="),
    options=[":maxdepth: 2"],
    entries=[
        example_entry("external_input.ipynb",
                      op_reference("fn.external_source", "Intro tutorial for external source")),
        example_entry(
            "parallel_external_source.ipynb",
            op_reference("fn.external_source", "How to use parallel mode for external source")),
        example_entry(
            "parallel_external_source_fork.ipynb",
            op_reference("fn.external_source",
                         "How to use parallel mode for external source in fork mode")),
        "dataloading_lmdb.ipynb",
        "dataloading_recordio.ipynb",
        "dataloading_tfrecord.ipynb",
        "dataloading_webdataset.ipynb",
        "coco_reader.ipynb",
        "numpy_reader.ipynb",
    ])
