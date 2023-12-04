doc(
    title="Image Processing",
    underline_char="=",
    entries=[
        doc_entry(
            "augmentation_gallery.ipynb",
            [
                op_reference("fn.erase", "Augmentation gallery"),
                op_reference("fn.water", "Augmentation gallery"),
                op_reference("fn.sphere", "Augmentation gallery"),
                op_reference("fn.warp_affine", "Augmentation gallery"),
                op_reference("fn.jpeg_compression_distortion", "Augmentation gallery"),
                op_reference("fn.paste", "Augmentation gallery"),
                op_reference("fn.flip", "Augmentation gallery"),
                op_reference("fn.rotate", "Augmentation gallery"),
                op_reference("fn.hsv", "Augmentation gallery"),
                op_reference("fn.brightness_contrast", "Augmentation gallery"),
            ],
        ),
        doc_entry(
            "brightness_contrast_example.ipynb",
            op_reference("fn.brightness_contrast", "BrightnessContrast example", 0),
        ),
        doc_entry(
            "color_space_conversion.ipynb",
            op_reference("fn.color_space_conversion", "Color space conversion tutorial", 0),
        ),
        doc_entry(
            "decoder_examples.ipynb",
            [
                op_reference("fn.decoders.image", "Image decoder examples", 0),
                op_reference("fn.decoders.image_random_crop", "Image decoder examples", 0),
                op_reference("fn.decoders.image_crop", "Image decoder examples", 0),
                op_reference("fn.decoders.image_slice", "Image decoder examples", 0),
            ],
        ),
        doc_entry(
            "hsv_example.ipynb",
            op_reference("fn.hsv", "HSV example", 0),
        ),
        doc_entry("interp_types.ipynb", op_reference("fn.resize", "Interpolation methods", 1)),
        doc_entry("resize.ipynb", op_reference("fn.resize", "Resize operator tutorial", 0)),
        doc_entry("warp.ipynb", op_reference("fn.warp_affine", "WarpAffine example", 0)),
        doc_entry(
            "3d_transforms.ipynb",
            [
                op_reference("fn.resize", "3D transforms", 3),
                op_reference("fn.warp_affine", "3D transforms"),
                op_reference("fn.rotate", "3D transforms"),
            ],
        ),
    ],
)
