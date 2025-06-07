# Image Processing Scripts

This folder contains scripts for handling image processing tasks.

## Scripts

### decode.py
Contains functionality for decoding images using NVIDIA DALI's decoders. The script:
- Uses mixed device decoding for optimal performance
- Outputs RGB format images
- Configures JPEG decoding with fancy upsampling enabled
- Uses standard IDCT (Inverse Discrete Cosine Transform) implementation

### decode_and_transform_from_json.py
Contains functionality for decoding and transforming images based on JSON configuration files. The script:
- Reads image and transformation parameters from JSON files using external source
- DALI pipeline decodes the image and crops it based on the transformation parameters and resizes to a given size
- Allows batch processing of multiple images

This example requires (COCO-Wholebody)[ https://github.com/jin-s13/COCO-WholeBody/] repostitory to be available in the `root_dir`.

### decode_and_transform_pytorch.py
Contains functionality for decoding and transforming images using PyTorch. The script:
- Uses PyTorch's data loading utilities to load images and labels from a JSON file
- DALI pipeline decodes the image and crops it based on the transformation parameters
- The multiprocessing environment requires the use of DALI proxy to run the pipeline

The json file should have the following format:
```
[
     {
       "image_id": "img0.jpg",
       "label": 0
     },
     {
       "image_id": "img11.jpg",
       "label": 1
     },
]
```
Where `image_id` is the name of the image file and `label` is the class label.

The image directory should have the following format:
```
img/
├── img0.jpg
├── img11.jpg
```