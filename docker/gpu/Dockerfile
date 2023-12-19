FROM tensorflow/tensorflow:1.15.0-gpu-py3

WORKDIR "/root"

COPY /docker/gpu/requirements.txt /root/
RUN pip3 install -r requirements.txt  && \
    rm requirements.txt

RUN mkdir -p /mask_rcnn_dev
WORKDIR /mask_rcnn_dev