FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libv4l-dev \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

RUN pip install opencv-python-headless paho-mqtt flask requests

COPY main.py /main.py
COPY web/ /web/

RUN mkdir -p /data

RUN printf '#!/bin/sh\npython3 /main.py\n' > /run.sh && chmod +x /run.sh