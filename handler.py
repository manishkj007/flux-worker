"""
RunPod Serverless handler — FLUX.1-schnell Text-to-Image.
Light worker: model weights loaded from RunPod network storage volume.
Deploy on RunPod as a serverless endpoint with an A40/L40S/A100 GPU.

Network storage setup:
  Mount your RunPod network volume at /runpod-volume. Model should be at:
    /runpod-volume/flux-schnell/

  To pre-download (one-time):
    from diffusers import FluxPipeline; import torch
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16)
    pipe.save_pretrained("/runpod-volume/flux-schnell")

Input:
  prompt: str          — image prompt
  width: int           — default 1024
  height: int          — default 1792 (9:16 for reels)
  num_inference_steps: int — default 4 (schnell is fast)
  seed: int            — optional reproducibility

Returns:
  image_base64: str    — PNG image as base64
  inference_time: float
"""
import os, time, base64, io, uuid
import runpod

pipe = None
device = "cuda"
_torch = None

# Network volume paths
VOLUME_BASE = os.environ.get("VOLUME_PATH", "/runpod-volume")
MODEL_PATH = os.path.join(VOLUME_BASE, "flux-schnell")
HF_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
HF_TOKEN = os.environ.get("HF_TOKEN", None)

def ensure_torch():
    global _torch, device
    if _torch is not None:
        return _torch
    import torch
    _torch = torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] torch={torch.__version__} cuda={torch.cuda.is_available()}")
    return torch

def load_model():
    global pipe
    if pipe is not None:
        return
    torch = ensure_torch()
    from diffusers import FluxPipeline

    # Prefer network volume, fallback to HF hub (and save to volume for next time)
    if os.path.isdir(MODEL_PATH) and os.path.isfile(os.path.join(MODEL_PATH, "model_index.json")):
        model_src = MODEL_PATH
        print(f"[flux] loading from network volume: {model_src}")
    else:
        model_src = HF_MODEL_ID
        print(f"[flux] volume not found at {MODEL_PATH}, downloading from {model_src}")

    kwargs = {"torch_dtype": torch.float16}
    if HF_TOKEN and model_src == HF_MODEL_ID:
        kwargs["token"] = HF_TOKEN
    pipe = FluxPipeline.from_pretrained(model_src, **kwargs)

    # Auto-save to network volume if downloaded from HF (for faster next cold start)
    if model_src == HF_MODEL_ID and os.path.isdir(VOLUME_BASE):
        try:
            print(f"[flux] saving to network volume: {MODEL_PATH}")
            pipe.save_pretrained(MODEL_PATH)
            print(f"[flux] saved successfully")
        except Exception as e:
            print(f"[flux] warning: could not save to volume: {e}")

    pipe.enable_model_cpu_offload()
    print(f"[flux] ready on {device}")

def handler(event):
    try:
        inp = event.get("input", {})
        prompt = inp.get("prompt", "")
        if not prompt:
            return {"error": "prompt is required"}

        load_model()

        width = int(inp.get("width", 1024))
        height = int(inp.get("height", 1792))
        steps = int(inp.get("num_inference_steps", 4))
        seed = inp.get("seed")

        generator = None
        if seed is not None:
            torch = ensure_torch()
            generator = torch.Generator(device=device).manual_seed(int(seed))

        t0 = time.time()
        print(f"[flux] gen {width}x{height} {steps}s")

        result = pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=generator,
        )

        image = result.images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        dt = time.time() - t0
        print(f"[flux] done {dt:.1f}s {len(b64)}B")

        return {
            "image_base64": b64,
            "inference_time": round(dt, 1),
            "width": width,
            "height": height,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
