FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
#from ubuntu:16.04

RUN echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDNN_VERSION 5.1.10
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
            libcudnn5=$CUDNN_VERSION-1+cuda8.0 \
            libcudnn5-dev=$CUDNN_VERSION-1+cuda8.0 && \
    rm -rf /var/lib/apt/lists/*

RUN cp /usr/lib/x86_64-linux-gnu/libcudnn* /usr/local/cuda-8.0/lib64
RUN cp /usr/include/cudnn.h /usr/local/cuda-8.0/include
RUN chmod a+r /usr/local/cuda-8.0/include/cudnn.h /usr/local/cuda-8.0/lib64/libcudnn*

# Run apt to install OS packages
RUN apt update
#RUN DEBIAN_FRONTEND=noninteractive apt-get install lightdm -y
RUN apt install -y tree vim curl 
#python3-pip
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:jonathonf/python-3.6
RUN apt-get update
RUN apt-get install -y python3.6 python3.6-dev python3-pip python3-venv 
#python3.6-venv
RUN apt-get install -y git
RUN apt-get install -y locales
RUN locale-gen en_US.UTF-8
# Download wget

RUN mkdir /usr/lib/nvidia
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-390
RUN apt-get install -y dkms nvidia-modprobe
# Python 3 package install example
RUN python3.6 -m pip install --upgrade pip 
#&& python3.6 -m pip install wheel
RUN ln -s /usr/bin/python3.6 /usr/local/bin/python3
#RUN ln -s /usr/local/bin/pip /usr/local/bin/pip3


#RUN pip3 install --upgrade pip
RUN python3.6 -m pip install http://download.pytorch.org/whl/cu80/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
RUN python3.6 -m pip install torchvision

#ENV PYTHONPATH="/usr/local/lib/python3.6/dist-packages"
RUN python3.6 -m pip install ipython matplotlib numpy pandas scikit-learn scipy six
RUN python3.6 -m pip install allennlp
RUN python3.6 -m pip install git+https://github.com/boudinfl/pke.git
RUN python3.6 -m pip install tqdm
RUN python3.6 -m pip install nltk


# create directory for "work".
RUN mkdir /work

# clone the rich context repo into /rich-context-competition
RUN git clone https://github.com/Coleridge-Initiative/rich-context-competition.git /rich-context-competition

LABEL maintainer="jonathan.morgan@nyu.edu"
