# FLUX.1-schnell — RunPod Serverless Worker

Light worker for [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) text-to-image generation on RunPod Serverless.

## Architecture

- **Docker image**: Code + dependencies only (~8GB). No model weights baked in.
- **Model weights**: Stored on RunPod Network Volume at `/runpod-volume/flux-schnell/` (~34GB).
- **GPU**: Requires AMPERE+ GPU with ≥24GB VRAM (A40, L40S, A100).

## Setup

### 1. One-time: Download model to network storage

SSH into a RunPod GPU pod with your network volume mounted, then run:

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.float16,
)
pipe.save_pretrained("/runpod-volume/flux-schnell")
```

### 2. Docker image (built automatically via GitHub Actions)

Push to `main` branch → GitHub Actions builds and pushes to DockerHub as `manish922020/flux-worker:latest`.

### 3. Create RunPod serverless endpoint

- Template: Use `manish922020/flux-worker:latest`
- Network Volume: Attach volume containing `/flux-schnell/`
- GPU: A40 or L40S recommended

## API

**Input:**
```json
{
  "prompt": "a futuristic cityscape at sunset",
  "width": 1024,
  "height": 1024,
  "num_inference_steps": 4,
  "seed": 42
}
```

**Output:**
```json
{
  "image_base64": "...",
  "inference_time": 3.2,
  "width": 1024,
  "height": 1024
}
```
