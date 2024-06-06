EfficientDet with TensorFlow and DALI
=====================================

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
	    --input_type tfrecord \
            --train_file_pattern './tfrecords/train*.tfrecord' \
            --batch_size 16 \
            --train_steps 2000 \
            --output_filename final_weights.h5

Evaluation in Keras Fit/Compile mode
------------------------------------
For the evaluation with DALI gpu pipeline::

    python3 eval.py \
            --pipeline dali_gpu \
	    --input_type tfrecord \
            --eval_file_pattern './tfrecords/eval*.tfrecord' \
            --eval_steps 5000 \
            --weights final_weights.h5

Usage
-----

.. code-block::

  usage: train.py [-h] [--initial_epoch INITIAL_EPOCH] [--epochs EPOCHS]
                  --input_type {tfrecord,coco} [--images_path IMAGES_PATH]
                  [--annotations_path ANNOTATIONS_PATH]
                  [--train_file_pattern TRAIN_FILE_PATTERN]
                  [--batch_size BATCH_SIZE] [--train_steps TRAIN_STEPS]
                  [--eval_file_pattern EVAL_FILE_PATTERN]
                  [--eval_steps EVAL_STEPS] [--eval_freq EVAL_FREQ]
                  [--eval_during_training] [--eval_after_training]
                  --pipeline_type {synthetic,tensorflow,dali_cpu,dali_gpu}
                  [--multi_gpu [MULTI_GPU [MULTI_GPU ...]]] [--seed SEED]
                  [--hparams HPARAMS] [--model_name MODEL_NAME]
                  [--output_filename OUTPUT_FILENAME]
                  [--start_weights START_WEIGHTS] [--log_dir LOG_DIR]
                  [--ckpt_dir CKPT_DIR]

  optional arguments:
    -h, --help            show this help message and exit
    --initial_epoch INITIAL_EPOCH
                          Epoch from which to start training.
    --epochs EPOCHS       Epoch on which training should finish.
    --input_type {tfrecord,coco}
                          Input type.
    --images_path IMAGES_PATH
                          Path to COCO images.
    --annotations_path ANNOTATIONS_PATH
                          Path to COCO annotations.
    --train_file_pattern TRAIN_FILE_PATTERN
                          TFrecord files glob pattern for files with training data.
    --batch_size BATCH_SIZE
    --train_steps TRAIN_STEPS
                          Number of steps (iterations) in each epoch.
    --eval_file_pattern EVAL_FILE_PATTERN
                          TFrecord files glob pattern for files with evaluation data,
                          defaults to `train_file_pattern` if not given.
    --eval_steps EVAL_STEPS
                          Number of examples to evaluate during each evaluation.
    --eval_freq EVAL_FREQ
                          During training evaluation frequency.
    --eval_during_training
                          Whether to run evaluation every `eval_freq` epochs.
    --eval_after_training
                          Whether to run evaluation after finished training.
    --pipeline_type {synthetic,tensorflow,dali_cpu,dali_gpu}
                          Pipeline type used while loading and preprocessing data.
                          One of: tensorflow – pipeline used in original
                          EfficientDet implementation on
                          https://github.com/google/automl/tree/master/efficientdet
                          synthetic – like `tensorflow` pipeline type but repeats
                          one batch endlessly dali_gpu – pipeline which uses
                          Nvidia Data Loading Library (DALI) to run part of data
                          preprocessing on GPUs to improve efficiency
                          dali_cpu – like `dali_gpu` pipeline type but restricted
                          to run only on CPU
    --multi_gpu [MULTI_GPU [MULTI_GPU ...]]
                          List of GPUs to use, if empty defaults to all visible GPUs.
    --seed SEED
    --hparams HPARAMS     String or filename with parameters.
    --model_name MODEL_NAME
    --output_filename OUTPUT_FILENAME
                          Filename for final weights to save.
    --start_weights START_WEIGHTS
    --log_dir LOG_DIR     Directory for tensorboard logs.
    --ckpt_dir CKPT_DIR   Directory for saving weights each step.

.. code-block::

  usage: eval.py [-h] --input_type {tfrecord,coco} [--images_path IMAGES_PATH]
                 [--annotations_path ANNOTATIONS_PATH]
                 [--eval_file_pattern EVAL_FILE_PATTERN]
                 [--eval_steps EVAL_STEPS]
                 --pipeline_type {synthetic,tensorflow,dali_cpu,dali_gpu}
                 [--weights WEIGHTS] [--model_name MODEL_NAME] [--hparams HPARAMS]

  optional arguments:
    -h, --help            show this help message and exit
    --input_type {tfrecord,coco}
                          Input type.
    --images_path IMAGES_PATH
                          Path to COCO images.
    --annotations_path ANNOTATIONS_PATH
                          Path to COCO annotations.
    --eval_file_pattern EVAL_FILE_PATTERN
                          TFrecord files glob pattern for files with evaluation data.
    --eval_steps EVAL_STEPS
                          Number of examples to evaluate.
    --pipeline_type {synthetic,tensorflow,dali_cpu,dali_gpu}
                          Pipeline type used while loading and preprocessing data.
                          One of: tensorflow – pipeline used in original
                          EfficientDet implementation on
                          https://github.com/google/automl/tree/master/efficientdet
                          synthetic – like `tensorflow` pipeline type but repeats
                          one batch endlessly dali_gpu – pipeline which uses
                          Nvidia Data Loading Library (DALI) to run part of data
                          preprocessing on GPUs to improve efficiency dali_cpu –
                          like `dali_gpu` pipeline type but restricted to run
                          only on CPU
    --weights WEIGHTS     Name of the file with model weights.
    --model_name MODEL_NAME
    --hparams HPARAMS     String or filename with parameters.

Requirements
~~~~~~~~~~~~
::

   pip install -r requirements.txt
