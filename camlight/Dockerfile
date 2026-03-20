FROM ghcr.io/home-assistant/aarch64-base-python:3.11

RUN apk add --no-cache \
    libgcc \
    libstdc++ \
    v4l-utils \
    ffmpeg

RUN pip3 install opencv-python-headless paho-mqtt flask requests

COPY main.py /main.py
COPY web/ /web/

RUN printf '#!/usr/bin/with-contenv bashio\npython3 /main.py\n' > /run.sh && chmod +x /run.sh
RUN mkdir -p /data