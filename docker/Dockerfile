FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

RUN apt-get update -y && \
    apt-get install -y sudo wget python3-pip zip unzip git curl && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip3 install -U pip && pip3 install albumentations Cython graphviz mlflow numpy opencv-python pandas PyYAML scikit-image scikit-learn scipy seaborn sklearn tensorboard timm torch==1.6.0 torchvision==0.7.0 tqdm pytorch-lightning

RUN git clone https://github.com/philferriere/cocoapi.git && \
    cd cocoapi/PythonAPI && \
    python3 setup.py build_ext install