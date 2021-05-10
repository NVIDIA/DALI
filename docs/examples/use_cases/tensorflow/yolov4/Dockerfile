FROM nvcr.io/nvidia/tensorflow:21.04-tf2-py3
WORKDIR yolov4
RUN python -m pip install --upgrade pip
RUN apt update
RUN apt install ffmpeg libsm6 libxext6  -y

# Reinstall cuda - for some reason some .so files are not present within the default instalation
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
RUN mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get install -y software-properties-common
RUN add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
RUN apt-get update
RUN apt-get -y install cuda
 
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110 nvidia-dali-tf-plugin-cuda110

COPY * ./

CMD python3 src/main.py 
