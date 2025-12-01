Temporal Shift Module Inference in PaddlePaddle
===============================================

This demo shows how to use DALI pipeline for video classification in PaddlePaddle.

The model used for this demo is `TSM: Temporal Shift Module for Efficient Video Understanding <https://arxiv.org/abs/1811.08383>`_.

It is trained on `kinetics400 <https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics>`_ and weights will be downloaded automatically.

For inference, videos should be resized to 300p and clipped to 10 second in length, which can be done with ``ffpmeg``.

Run the following commands to download and preprocess some videos from kinetics400 valset.

.. code-block:: bash

   mkdir demo
   youtube-dl --quiet --no-warnings -f mp4 -o demo/tmp.mp4 \
              'https://www.youtube.com/watch?v=iU3ByohkPaM'
   ffmpeg -y -i demo/tmp.mp4 -filter:v scale=-1:300 -ss 0 -t 10 -c:a copy demo/1.mp4
   youtube-dl --quiet --no-warnings -f mp4 -o demo/tmp.mp4 \
              'https://www.youtube.com/watch?v=C0J6EQYYLzI'
   ffmpeg -y -i demo/tmp.mp4 -filter:v scale=-1:300 -ss 0 -t 10 -c:a copy demo/2.mp4
   rm demo/tmp.mp4

The script will extract 8 frames from the input videos with a stride of ``s`` (30 by default), and will output top ``k`` predicted (1 by default) labels for each video.

.. code-block:: bash

   python infer.py -k 1 -s 30 demo
   # will output
   # prediction for demo/1.mp4 is: ['carving_pumpkin']
   # prediction for demo/2.mp4 is: ['blowing_out_candles']


Requirements
------------

- Install the following python packages via pip or other means.

  - `PaddlePaddle <https://www.paddlepaddle.org>`_ (1.6 or above)

  - `Nvidia DALI <https://github.com/NVIDIA/DALI>`_

- Optionally, the following programs are needed for preparing the input videos.

  - `youtube-dl <https://github.com/ytdl-org/youtube-dl>`_

  - `ffmpeg <https://www.ffmpeg.org/>`_


Usage
-----

.. code-block:: bash

   usage: infer.py [-h] [--topk K] [--stride S] DIR

   Paddle Temporal Shift Module Inference

   positional arguments:
     DIR               Path to video files

   optional arguments:
     -h, --help        show this help message and exit
     --topk K, -k K    Top k results (default: 1)
     --stride S, -s S  Distance between frames (default: 30)
