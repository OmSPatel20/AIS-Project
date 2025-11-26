# Dockerfile (GPU) - requires nvidia-container-runtime / toolkit on host
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Make python3 default for pip commands
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:app"]
