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
- DALI pipeline decodes the image and crops it based on the transformation parameters, resizes to a given size and randomly flips
- Allows batch processing of multiple images

This example requires [DALI_extra](https://github.com/NVIDIA/DALI_extra/) repostitory to be available on a local machine.

### decode_and_transform_pytorch.py
Contains functionality for decoding and transforming images using PyTorch. The script:
- Uses PyTorch's data loading utilities to load images and additional landmark information saved as a numpy array
- DALI pipeline decodes the image and resizes it to a given size
- Returns transformed image and a corresponding landmark for each iteration
- The multiprocessing environment requires the use of DALI proxy to run the pipeline

This example requires [DALI_extra](https://github.com/NVIDIA/DALI_extra/) repostitory to be available on a local machine.
