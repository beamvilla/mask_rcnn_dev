#!/bin/bash
pip install virtualenv
virtualenv myenv
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local
conda install -q -y --prefix /usr/local python=3.7 ujson
conda install -y -c cudatoolkit=10.1
cp /content/drive/MyDrive/libcudnn7_7.6.5.32-1+cuda10.0_amd64.deb .
sudo dpkg -i /content/drive/MyDrive/libcudnn7_7.6.5.32-1+cuda10.1_amd64.deb
python3 /content/mask_rcnn_dev/activate_conda.py
python --version
sudo apt install unzip
pip install tensorflow-gpu==1.14.0 keras==2.2.5 scikit-image==0.16.2 h5py==2.10.0 matplotlib datetime pytest-shutil pandas ipykernel opencv-python protobuf==3.20.0