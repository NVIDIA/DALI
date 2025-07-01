# Video Processing Scripts

This folder contains scripts for handling video processing tasks.

## Scripts

### decode.py
Contains functionality for decoding videos using NVIDIA DALI's decoders. The script:
- Uses mixed device decoding
- Decodes video frames in sequence (default 30 frames)
- Resizes output to 1280x720 resolution
- Includes vertical flip transformation
- Saves sample frames as JPEG images

### decode_and_transform_from_json.py 
Contains functionality for decoding and transforming videos based on JSON configuration files. The script:
- Reads video files and clip timing information from JSON files
- Converts millisecond timestamps to frame numbers based on video FPS
- DALI pipeline decodes specific frame ranges from the videos
- Resizes output frames to 640x480 resolution
- Processes videos in batches
- Saves individual frames as JPEG images

This example requires [DomainSpecificHighlight](https://github.com/aliensunmin/DomainSpecificHighlight/) repostitory to be available in the `root_dir`.

### decode_and_transform_pytorch.py 
Contains functionality for decoding and transforming videos integrating PyTorch and DALI. The script:
- Reads video files using VideosDataset which implement Pytorch Dataset
- Uses DALI proxy to enable multiprocessing pipeline execution
- Decdes video files with DALI decoder
- Resizes output frames to 1280x720 resolution

This example requires [DALI_extra](https://github.com/NVIDIA/DALI_extra/) repostitory to be available on a local machine.
