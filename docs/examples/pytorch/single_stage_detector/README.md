# 1. Problem
Object detection.

# 2. Directions

### Steps to configure machine
From Source

Standard script.

From Docker
1. Checkout the MLPerf repository
```
git clone https://github.com/mlperf/reference.git
```
2. Install CUDA and Docker
```
source reference/install_cuda_docker.sh
```
3. Build the docker image for the single stage detection task
```
# Build from Dockerfile
cd reference/single_stage_detector/
sudo docker build -t mlperf/single_stage_detector .
```

### Steps to download data
```
cd reference/single_stage_detector/
source download_dataset.sh
```

### Steps to run benchmark.
From Source

Run the run_and_time.sh script
```
cd reference/single_stage_detector/ssd
source run_and_time.sh SEED TARGET
```
where SEED is the random seed for a run, TARGET is the quality target from Section 5 below.

Docker Image
```
sudo nvidia-docker run -v /coco:/coco -t -i --rm --ipc=host mlperf/single_stage_detector ./run_and_time.sh SEED TARGET
```

# 3. Dataset/Environment
### Publiction/Attribution.
Microsoft COCO: COmmon Objects in Context. 2017.

### Training and test data separation
Train on 2017 COCO train data set, compute mAP on 2017 COCO val data set.

# 4. Model.
### Publication/Attribution
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector. In the Proceedings of the European Conference on Computer Vision (ECCV), 2016.

Backbone is ResNet34 pretrained on ILSVRC 2012 (from torchvision). Modifications to the backbone networks: remove conv_5x residual blocks, change the first 3x3 convolution of the conv_4x block from stride 2 to stride1 (this increases the resolution of the feature map to which detector heads are attached), attach all 6 detector heads to the output of the last conv_4x residual block. Thus detections are attached to 38x38, 19x19, 10x10, 5x5, 3x3, and 1x1 feature maps. Convolutions in the detector layers are followed by batch normalization layers.

# 5. Quality.
### Quality metric
Metric is COCO box mAP (averaged over IoU of 0.5:0.95), computed over 2017 COCO val data.

### Quality target
mAP of 0.212

### Evaluation frequency

### Evaluation thoroughness
All the images in COCO 2017 val data set.
