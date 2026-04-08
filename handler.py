"""
RunPod Serverless handler — FLUX.1-schnell Text-to-Image.
Light worker: model weights loaded from RunPod network storage volume.
"""
import os, time, base64, io, shutil, subprocess

# ── Force ALL caches/temps to network volume BEFORE any other imports ──
VOLUME_BASE = os.environ.get("VOLUME_PATH", "/runpod-volume")
_vol_cache = os.path.join(VOLUME_BASE, "cache")
_vol_tmp = os.path.join(VOLUME_BASE, "tmp")

for d in [_vol_cache, _vol_tmp]:
    os.makedirs(d, exist_ok=True)

# Redirect every known cache/temp env var to the volume
os.environ["HF_HOME"] = _vol_cache
os.environ["HF_HUB_CACHE"] = os.path.join(_vol_cache, "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(_vol_cache, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(_vol_cache, "transformers")
os.environ["TORCH_HOME"] = os.path.join(_vol_cache, "torch")
os.environ["XDG_CACHE_HOME"] = _vol_cache
os.environ["TMPDIR"] = _vol_tmp
os.environ["TEMP"] = _vol_tmp
os.environ["TMP"] = _vol_tmp

import runpod

pipe = None
device = "cuda"
_torch = None

MODEL_PATH = os.path.join(VOLUME_BASE, "flux-schnell")
HF_MODEL_ID = "black-forest-labs/FLUX.1-schnell"
HF_TOKEN = os.environ.get("HF_TOKEN", None)


def disk_usage():
    """Print disk usage for debugging."""
    try:
        for path in ["/", VOLUME_BASE]:
            st = os.statvfs(path)
            total = st.f_blocks * st.f_frsize / (1024**3)
            free = st.f_bfree * st.f_frsize / (1024**3)
            used = total - free
            print(f"[disk] {path}: {used:.1f}G/{total:.1f}G used ({free:.1f}G free)")
        # List top-level volume contents  
        if os.path.isdir(VOLUME_BASE):
            for entry in os.listdir(VOLUME_BASE):
                fp = os.path.join(VOLUME_BASE, entry)
                if os.path.isdir(fp):
                    try:
                        sz = subprocess.check_output(["du", "-sh", fp], timeout=10).decode().split()[0]
                    except Exception:
                        sz = "?"
                    print(f"[disk]   {entry}/ => {sz}")
    except Exception as e:
        print(f"[disk] error: {e}")


def cleanup_old_cache():
    """Remove old HF cache and tmp to free space on the volume."""
    for name in ["hf_cache", "tmp"]:
        old = os.path.join(VOLUME_BASE, name)
        if os.path.isdir(old):
            sz = "?"
            try:
                sz = subprocess.check_output(["du", "-sh", old], timeout=30).decode().split()[0]
            except Exception:
                pass
            print(f"[cleanup] removing old {old} ({sz})")
            shutil.rmtree(old, ignore_errors=True)


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
    
    print("[flux] === starting model load ===")
    disk_usage()
    cleanup_old_cache()
    disk_usage()
    
    torch = ensure_torch()
    from diffusers import FluxPipeline

    # Check if model is already on the volume
    model_index = os.path.join(MODEL_PATH, "model_index.json")
    if os.path.isdir(MODEL_PATH) and os.path.isfile(model_index):
        print(f"[flux] loading from network volume: {MODEL_PATH}")
        pipe = FluxPipeline.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, local_files_only=True
        )
    else:
        print(f"[flux] model not on volume, downloading to {MODEL_PATH}...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=HF_MODEL_ID,
            local_dir=MODEL_PATH,
            token=HF_TOKEN,
            ignore_patterns=["*.md", "*.txt", ".gitattributes"],
        )
        print(f"[flux] download complete, loading model...")
        disk_usage()
        pipe = FluxPipeline.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, local_files_only=True
        )

    pipe.enable_model_cpu_offload()
    print(f"[flux] ready on {device}")
    disk_usage()

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
