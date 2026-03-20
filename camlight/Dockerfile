FROM python:3.11-alpine

RUN apk add --no-cache \
    libgcc \
    libstdc++ \
    v4l-utils \
    ffmpeg \
    dos2unix \
    musl-dev \
    gcc

RUN pip3 install --no-cache-dir opencv-python-headless paho-mqtt flask requests

COPY run.sh /run.sh
RUN dos2unix /run.sh && chmod +x /run.sh

COPY main.py /main.py
COPY web/ /web/

RUN mkdir -p /data