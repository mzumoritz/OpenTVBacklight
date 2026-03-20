FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    v4l-utils \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir opencv-python-headless paho-mqtt flask requests

COPY run.sh /run.sh
RUN sed -i 's/\r//' /run.sh && chmod +x /run.sh

COPY main.py /main.py
COPY web/ /web/
RUN mkdir -p /data