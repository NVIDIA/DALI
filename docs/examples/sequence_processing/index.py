doc(title="Video Processing",
    underline_char="=",
    entries=[
        doc_entry("video/video_reader_simple_example.ipynb",
                  op_reference('fn.readers.video', "Tutorial describing how to use video reader")),
        doc_entry("video/video_reader_label_example.ipynb",
                  op_reference('fn.readers.video', "Tutorial describing how to use video reader to output frames with labels")),
        doc_entry("video/video_file_list_outputs.ipynb",
                  op_reference('fn.readers.video', "Tutorial describing how output to frames with \
                                lables assigned to dedicated ranges of frame numbers/timestamps")),
        doc_entry("sequence_reader_simple_example.ipynb",
                  op_reference('fn.readers.sequence', "Tutorial describing how to read sequence of video frames stores as separete files")),
        doc_entry("optical_flow_example.ipynb",
                  [op_reference('fn.readers.video', "Tutorial describing how read video and calculate optical flow"),
                   op_reference('fn.optical_flow', "Tutorial describing how to calculate optical flow from a sequence of frames")]),
    ])
