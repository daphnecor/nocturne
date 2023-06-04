FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

MAINTAINER "Daphne Cornelisse <cornelisse.daphne@nyu.edu>"

ENV SFML_DIR /usr/share/SFML

RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    git \
    libsfml-dev \
    python3-dev \
    python3-pip \
    python3.10-venv \
    && apt upgrade -y \
    && apt-get -y clean

RUN pip3 install -U pip