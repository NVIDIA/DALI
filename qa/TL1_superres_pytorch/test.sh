#!/bin/bash -e
pip_packages='pillow numpy torch torchvision scikit-image tensorboardX protobuf<4'
target_dir=./docs/examples/use_cases/video_superres

do_once() {
    apt-get update
    apt-get install -y wget ffmpeg git

    mkdir -p video_files

    container_path=${DALI_EXTRA_PATH}/db/optical_flow/sintel_trailer/sintel_trailer.mp4

    IFS='/' read -a container_name <<< "$container_path"
    IFS='.' read -a split <<< "${container_name[-1]}"

    for i in {0..4};
    do
        ffmpeg -ss 00:00:${i}0 -t 00:00:10 -i $container_path -vcodec copy -acodec copy -y video_files/${split[0]}_$i.${split[1]}
    done

    DATA_DIR=data_dir/720p/scenes
    # Creating simple working env for PyTorch SuperRes example
    mkdir -p $DATA_DIR/train/
    mkdir -p $DATA_DIR/val/


    cp video_files/* $DATA_DIR/train/
    cp video_files/* $DATA_DIR/val/

    # Pre-trained FlowNet2.0 weights
    # publicly available on https://drive.google.com/file/d/1QW03eyYG_vD-dT-Mx4wopYvtPu_msTKn/view
    FLOWNET_PATH=/dali_internal/FlowNet2-SD_checkpoint.pth.tar

    git clone https://github.com/NVIDIA/flownet2-pytorch.git
    cd flownet2-pytorch
    git checkout 6a0d9e70a5dcc37ef5577366a5163584fd7b4375
    cd ..

}

test_body() {

    python main.py --loader DALI --rank 0 --batchsize 2 --frames 3 --root $DATA_DIR --world_size 1 --is_cropped --max_iter 100 --min_lr 0.0001 --max_lr 0.001 --crop_size 512 960 --flownet_path $FLOWNET_PATH

    cd ..
}

pushd ../..
source ./qa/test_template.sh
popd
