FROM ghcr.io/home-assistant/aarch64-base-python:3.11

RUN apk add --no-cache \
    libgcc \
    libstdc++ \
    v4l-utils \
    ffmpeg \
    dos2unix

RUN pip3 install opencv-python-headless paho-mqtt flask requests

COPY run.sh /run.sh
RUN dos2unix /run.sh && chmod +x /run.sh

COPY main.py /main.py
COPY web/ /web/

RUN mkdir -p /data