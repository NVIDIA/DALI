FROM nvcr.io/nvidia/tensorflow:21.04-tf2-py3
WORKDIR yolov4

RUN apt clean autoclean

RUN python -m pip install --upgrade pip
RUN apt update
RUN apt install ffmpeg libsm6 libxext6  -y

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110 nvidia-dali-tf-plugin-cuda110

COPY * ./

CMD python3 src/main.py
