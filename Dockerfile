FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN echo "XKBMODEL=\"pc105\"\n \
          XKBLAYOUT=\"us\"\n \
          XKBVARIANT=\"\"\n \
          XKBOPTIONS=\"\"" > /etc/default/keyboard

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libnccl2=2.1.4-1+cuda9.0 \
         libnccl-dev=2.1.4-1+cuda9.0 \
         libjpeg-dev \
         libpng-dev \
         sudo \
         apt-utils \
         man \
         tmux \
         less \
         wget \
         iputils-ping \
         zsh \
         htop \
         software-properties-common \
         tzdata \
         locales \
         openssh-server \
         xauth \
         rsync &&\
     rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8
ENV LANG="en_US.UTF-8" LANGUAGE="en_US:en" LC_ALL="en_US.UTF-8"

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \     
     rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda install -y -c pytorch pytorch torchvision magma-cuda80 && \
    conda clean -ya 

RUN echo "export PATH=/opt/conda/bin:\$PATH" > /etc/profile.d/conda.sh
RUN echo "export PATH=/usr/local/nvidia/bin:\$PATH" > /etc/profile.d/nvidia.sh

RUN echo "X11UseLocalhost no" >> /etc/ssh/sshd_config
ENV PYTHONUNBUFFERED=1
WORKDIR /root
EXPOSE 22 80 443 8000
ENTRYPOINT service ssh restart && ldconfig && bash
