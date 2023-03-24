FROM tensorflow/tensorflow:1.15.0-gpu-py3

WORKDIR /root

ADD /requirements.txt /root/requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

RUN mkdir -p /mask_rcnn_dev
WORKDIR /mask_rcnn_dev