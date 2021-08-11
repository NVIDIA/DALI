ARG BASE_VERSION=21.05
FROM nvcr.io/nvidia/tensorflow:${BASE_VERSION}-tf2-py3
WORKDIR effdet
RUN python -m pip install --upgrade pip
RUN apt update
RUN apt install ffmpeg libsm6 libxext6  -y

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .

CMD python3 train.py 
