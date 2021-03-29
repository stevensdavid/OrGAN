# used these as examples:
# https://github.com/kaust-vislab/python-data-science-project/blob/master/docker/Dockerfile
# https://hub.docker.com/r/anibali/pytorch/dockerfile
# https://github.com/pytorch/pytorch/blob/master/docker/pytorch/Dockerfile
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04

SHELL [ "/bin/bash", "--login", "-c" ]

# install utilities
RUN apt-get update && \ 
   DEBIAN_FRONTEND="noninteractive" \ 
   apt-get install -y --no-install-recommends \
   build-essential \
   cmake \
   git \
   curl \
   ca-certificates \
   sudo \
   bzip2 \
   libx11-6 \
   wget \
   vim \
   libjpeg-dev \
   libpng-dev && \
   rm -rf /var/lib/apt/lists/*

# Create a non-root user
# change your uid (run id -u to learn it) 
# and gid (run id -g to learn it)
ARG username=dstevens
ARG uid=1015
ARG gid=1015
ENV USER $username
ENV UID $uid
ENV GID $gid
ENV HOME /home/$USER

RUN addgroup --gid $GID $USER
RUN adduser --disabled-password \
   --gecos "Non-root user" \
   --uid $UID \
   --gid $GID \
   --home $HOME \
   $USER

# switch to that user
USER $USER

# install miniconda, latest as of writing
ENV MINICONDA_VERSION py39_4.9.2

ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
   chmod +x ~/miniconda.sh && \
   ~/miniconda.sh -b -p $CONDA_DIR && \
   rm ~/miniconda.sh

# add conda to path (so that we can just use conda install <package> in the rest of the dockerfile)
ENV PATH=$CONDA_DIR/bin:$PATH

# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile

# make conda activate command available from /bin/bash --interactive shells
RUN conda init bash

# create a project directory inside user home
# you will login to this point, or jupyter will run from this point
ENV PROJECT_DIR $HOME/app
RUN mkdir $PROJECT_DIR
WORKDIR $PROJECT_DIR

COPY environment.yml ${PROJECT_DIR}/environment.yml

ENV SHELL=/bin/bash

# build the conda environment
ENV ENV_PREFIX $PROJECT_DIR/env
RUN conda update --name base --channel defaults conda && \
   conda env create -f environment.yml && \
   conda clean --all --yes

# Add code
# COPY --chown=${USER}:${GID} . ${PROJECT_DIR}
RUN git clone https://github.com/stevensdavid/msc.git