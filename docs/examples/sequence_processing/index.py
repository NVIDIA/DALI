doc(title="Video Processing",
    underline_char="=",
    entries=[
        doc_entry("video/video_reader_simple_example.ipynb",
                  op_reference('fn.readers.video', "Tutorial describing how to use video reader")),
        doc_entry("video/video_reader_label_example.ipynb",
                  op_reference('fn.readers.video', "Tutorial describing how to use video reader to output frames with labels")),
        doc_entry("video/video_file_list_outputs.ipynb",
                  op_reference('fn.readers.video', "Tutorial describing how to output frames with \
                                labels assigned to dedicated ranges of frame numbers/timestamps")),
        doc_entry("sequence_reader_simple_example.ipynb",
                  op_reference('fn.readers.sequence', "Tutorial describing how to read sequence of video frames stored as separate files")),
        doc_entry("optical_flow_example.ipynb",
                  [op_reference('fn.readers.video', "Tutorial describing how to calculate optical flow from video inputs"),
                   op_reference('fn.optical_flow', "Tutorial describing how to calculate optical flow from sequence inputs")]),
    ])
