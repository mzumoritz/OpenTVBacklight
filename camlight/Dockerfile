FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libv4l-dev \
    v4l-utils \
    bash \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install opencv-python-headless paho-mqtt flask requests

RUN curl -fsSL https://raw.githubusercontent.com/hassio-addons/bashio/main/lib/bashio.sh \
    -o /usr/local/bin/bashio && chmod +x /usr/local/bin/bashio

COPY run.sh /run.sh
COPY main.py /main.py
COPY web/ /web/

RUN chmod +x /run.sh
RUN mkdir -p /data

CMD ["/run.sh"]