#!/bin/bash

###################
# TRAINING CONFIG #
###################

export DATA_DIR=data_dir
export RESOLUTION=720p  # options: 540p, 720p, 1080p, 4K
export LOADER="DALI"  # options: "DALI" or "pytorch"
export DATA_TYPE=scenes # options: "scenes" or "frames"
#export CODEC="h264"  #
#export CRF="18"      # set these three only if used during preprocessing
#export KEYINT="4"    #
export ROOT=$DATA_DIR/$RESOLUTION/$DATA_TYPE
export IS_CROPPED="--is_cropped"  # Uncomment to crop input images
#export CROP_SIZE="-1 -1"
export CROP_SIZE="512 960"  # Only applicable if --is_cropped uncommented
export TIMING="--timing"  # Uncomment to time data loading and computation - slower
#export FP16="--fp16"  # Uncomment to load data and train model in fp16
export MINLR=0.0001
export MAXLR=0.001
export BATCHSIZE=2
export FRAMES=3
export MAX_ITER=320000
export WS=1  # Number of GPUs available
export BASE_RANK=0  # Device ID of first GPU (assumes GPUs numbered sequentially)

export IP=localhost

# tensorboard --logdir runs 2> /dev/null &
# echo "Tensorboard launched"

python main.py --loader $LOADER --rank $(($BASE_RANK)) --batchsize $BATCHSIZE --frames $FRAMES --root $ROOT --world_size $WS --ip $IP $BENCHMARK $IS_CROPPED $FP16 --max_iter $MAX_ITER --min_lr $MINLR --max_lr $MAXLR $TIMING --crop_size $CROP_SIZE
