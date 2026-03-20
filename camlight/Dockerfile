ARG BUILD_FROM
FROM $BUILD_FROM

RUN apk add --no-cache \
    libgcc \
    libstdc++ \
    v4l-utils \
    ffmpeg

RUN pip install opencv-python-headless paho-mqtt flask requests

COPY main.py /main.py
COPY web/ /web/
COPY run.sh /run.sh

RUN chmod +x /run.sh
RUN mkdir -p /data