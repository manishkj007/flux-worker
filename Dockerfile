# ── RunPod Serverless — FLUX.1-schnell Text-to-Image (Light Worker) ──────
# Model weights stored on RunPod Network Volume (/runpod-volume/flux-schnell/).
# Image is ~8GB (code + deps only, no model weights baked in).
#
# On network volume:
#   /runpod-volume/flux-schnell/  — FLUX.1-schnell model (~34GB)
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir \
    diffusers transformers accelerate sentencepiece protobuf runpod

COPY handler.py /app/handler.py

CMD ["python3", "/app/handler.py"]
