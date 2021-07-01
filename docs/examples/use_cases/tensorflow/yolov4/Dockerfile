FROM nvcr.io/nvidia/tensorflow:21.06-tf2-py3
WORKDIR yolov4

RUN apt clean autoclean
RUN python -m pip install --upgrade pip
RUN apt update
RUN apt install python3-opencv  -y

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY * ./
