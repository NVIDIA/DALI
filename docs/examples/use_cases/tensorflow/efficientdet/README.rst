EfficientDet with TensorFlow and DALI
=================================

This is a modified version of original EfficientDet implementation
https://github.com/google/automl/tree/master/efficientdet.
It has been changed to allow to use DALI data preprocessing.

To use DALI pipeline for data loading and preprocessing ``--pipeline dali_gpu`` or
``--pipeline dali_cpu``, for original pipeline ``--pipeline tensorflow``.

Preparing data files from COCO dataset
--------------------------------------
For creating TFrecords files::

    python3 ./dataset/create_coco_tfrecord.py \
            --image_dir ./coco/train2017 \
            --object_annotations_file ./coco/annotations/instances_train2017.json \
            --output_file_prefix ./tfrecords/train

For creating TFrecord index files (necessary only for DALI pipelines)::

    python3 ./dataset/create_tfrecord_indexes.py \
            --tfrecord_file_pattern './tfrecords/*.tfrecord' \
            --tfrecord2idx_script ../../../../../tools/tfrecord2idx \

Training in Keras Fit/Compile mode
----------------------------------
For the full training on all available GPUs with DALI gpu pipeline::

    python3 train.py \
            --multi_gpu \
            --pipeline dali_gpu \
            --epochs 50 \
            --train_file_pattern './tfrecords/train*.tfrecord' \
            --train_batch_size 16 \
            --train_steps 2000 \
            --output final_weights.h5

Evaluation in Keras Fit/Compile mode
------------------------------------
For the evaluation with DALI gpu pipeline::

    python3 eval.py \
            --pipeline dali_gpu \
            --eval_file_pattern './tfrecords/eval*.tfrecord' \
            --eval_steps 5000 \
            --weights final_weights.h5

Usage
-----

.. code-block::

  usage: train.py [-h] [--initial_epoch INITIAL_EPOCH] [--epochs EPOCHS]
                  --train_file_pattern TRAIN_FILE_PATTERN
                  [--train_batch_size TRAIN_BATCH_SIZE]
                  [--train_steps TRAIN_STEPS]
                  [--eval_file_pattern EVAL_FILE_PATTERN]
                  [--eval_steps EVAL_STEPS] [--eval_freq EVAL_FREQ]
                  [--eval_during_training] [--eval_after_training] --pipeline
                  {syntetic,tensorflow,dali_cpu,dali_gpu}
                  [--multi_gpu [MULTI_GPU [MULTI_GPU ...]]] [--seed SEED]
                  [--hparams HPARAMS] [--model_name MODEL_NAME]
                  [--output OUTPUT] [--start_weights START_WEIGHTS]
                  [--log_dir LOG_DIR] [--ckpt_dir CKPT_DIR]

  optional arguments:
    -h, --help            show this help message and exit
    --initial_epoch INITIAL_EPOCH
                          epoch from which to start training
    --epochs EPOCHS       epoch on which training should finish
    --train_file_pattern TRAIN_FILE_PATTERN
                          glob pattern for TFrecord files with training data
    --train_batch_size TRAIN_BATCH_SIZE
    --train_steps TRAIN_STEPS
                          number of steps in each epoch
    --eval_file_pattern EVAL_FILE_PATTERN
                          glob pattern for TFrecord files with evaluation data,
                          defaults to TRAIN_FILE_PATTERN if not given
    --eval_steps EVAL_STEPS
                          number of examples to evaluate
    --eval_freq EVAL_FREQ
                          during training evalutaion frequency
    --eval_during_training
                          whether to run evaluation every EVAL_FREQ epochs
    --eval_after_training
                          whether to run evaluation after finished training
    --pipeline {syntetic,tensorflow,dali_cpu,dali_gpu}
                          pipeline type
    --multi_gpu [MULTI_GPU [MULTI_GPU ...]]
                          list of GPUs to use, defaults to all visible GPUs
    --seed SEED
    --hparams HPARAMS     string or filename with parameters
    --model_name MODEL_NAME
    --output OUTPUT       filename for final weights to save
    --start_weights START_WEIGHTS
    --log_dir LOG_DIR     directory for tensorboard logs
    --ckpt_dir CKPT_DIR   directory for saving weights each step

.. code-block:: 

  usage: eval.py [-h] --eval_file_pattern EVAL_FILE_PATTERN
                 [--eval_steps EVAL_STEPS] --pipeline
                 {syntetic,tensorflow,dali_cpu,dali_gpu} [--weights WEIGHTS]
                 [--model_name MODEL_NAME] [--hparams HPARAMS]

  optional arguments:
    -h, --help            show this help message and exit
    --eval_file_pattern EVAL_FILE_PATTERN
                          glob pattern for TFrecord files with evaluation data
    --eval_steps EVAL_STEPS
                          number of examples to evaluate
    --pipeline {syntetic,tensorflow,dali_cpu,dali_gpu}
                          pipeline type
    --weights WEIGHTS     file with model weights
    --model_name MODEL_NAME
    --hparams HPARAMS     string or filename with parameters


Requirements
~~~~~~~~~~~~
::

   pip install -r requirements.txt
