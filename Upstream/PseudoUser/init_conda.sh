#!/bin/bash

# ----  set your username ---- #
USERNAME="LLM"

mkdir -p /data1/$USERNAME/miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /data1/$USERNAME/miniconda/miniconda.sh

bash /data1/$USERNAME/miniconda/miniconda.sh -b -u -p /data1/$USERNAME/miniconda

rm -rf /data1/$USERNAME/miniconda/miniconda.sh

/data1/$USERNAME/miniconda/bin/conda init bash

source ~/.bashrc

echo "export CONDA_PKGS_DIRS=/data1/$USERNAME/miniconda/pkgs" >> ~/.bashrc
echo "export PYTHONUSERBASE=/data1/$USERNAME/.local" >> ~/.bashrc
echo "export PIP_CACHE_DIR=/data1/$USERNAME/.cache/pip" >> ~/.bashrc

mkdir -p /data1/$USERNAME/miniconda/pkgs
mkdir -p /data1/$USERNAME/.local
mkdir -p /data1/$USERNAME/.cache/pip

source ~/.bashrc