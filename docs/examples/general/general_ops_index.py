doc(title="General Purpose",
    underline_char="=",
    entries=[
        "expressions/index.py",
        doc_entry(
            "reductions.ipynb",
            op_reference('fn.reductions', "Tutorial describing how to use reductions")),    
        doc_entry(
            "tensor_join.ipynb",
            [
                op_reference('fn.cat', "Tutorial describing tensor joining"),
                op_reference('fn.stack', "Tutorial describing tensor joining")]),
        doc_entry(
            "reinterpret.ipynb",
            [
                op_reference('fn.reshape', "Tutorial describing tensor reshaping"),
                op_reference('fn.squeeze', "Tutorial describing tensor squeezing"),
                op_reference('fn.expand_dims', "Tutorial describing tensor dimensions expanding"),
                op_reference('fn.reinterpret', "Tutorial describing tensor reinterpreting")]),
        doc_entry(
            "normalize.ipynb",
            op_reference('fn.normalize', "Tutorial describing tensor normalization")),
        doc_entry(
            "../math/geometric_transforms.ipynb",
            [
                op_reference('fn.transforms', "Tutorial describing tensor geometric transformations to transform points and images"),
                op_reference('fn.warp_affine', "Tutorial showing how to use afine transform"),
                op_reference('fn.coord_transform', "Tutorial describing how to transform points accompanying images")]),
        doc_entry(
            "erase.ipynb",
            op_reference('fn.erase', "Tutorial describing tensor erasing"))
    ])
