FROM tensorflow/tensorflow:1.15.3-py3
WORKDIR /root

RUN apt-get update --yes --quiet \
&& apt-get --yes --quiet upgrade \
&& apt-get install --yes --quiet --no-install-recommends gcc \
&& rm -rf /var/lib/apt/lists/*

ADD /docker/requirements.txt /root/requirements.txt
RUN pip3 install --user --no-cache-dir -r requirements.txt

RUN mkdir -p /mask_rcnn_dev
WORKDIR /mask_rcnn_dev