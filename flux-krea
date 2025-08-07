from io import BytesIO
from pathlib import Path
import os
from typing import Optional

import modal
import torch
from diffusers import FluxPipeline
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response
from PIL import Image

# Ustawienia środowiska
diffusers_commit_sha = "00f95b9755718aabb65456e791b8408526ae6e76"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        f"uv pip install --system --compile-bytecode --index-strategy unsafe-best-match "
        f"accelerate~=1.8.1 git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha} "
        f"huggingface-hub[hf-transfer]~=0.33.1 Pillow safetensors transformers sentencepiece torch==2.7.1 fastapi[standard]==0.115.4 python-multipart"
    )
)

MODEL_NAME = "black-forest-labs/FLUX.1-Krea-dev"

app = modal.App("flux-krea-fastapi")

@app.cls(image=image, gpu="L40s", timeout=10 * 60)
class FluxKreaModel:
    def __enter__(self):
        print(f"Loading {MODEL_NAME}...")
        self.pipe = FluxPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
    @modal.method()
    def inference(self, prompt: str, width: int = 1024, height: int = 1024,
                  guidance_scale: float = 7.5, num_inference_steps: int = 30,
                  seed: Optional[int] = None) -> bytes:
        generator = torch.Generator("cuda").manual_seed(seed) if seed else torch.Generator("cuda")
        image = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    web_app = FastAPI()
    @web_app.post("/text_to_image")
    async def text_to_image(
        prompt: str = Form(...),
        width: int = Form(1024),
        height: int = Form(1024),
        guidance_scale: float = Form(7.5),
        num_inference_steps: int = Form(30),
        seed: Optional[int] = Form(None),
    ):
        try:
            model = FluxKreaModel()
            result = model.inference.remote(
                prompt=prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
            )
            return Response(content=result, media_type="image/png")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return web_app

if __name__ == "__main__":
    modal run qwen_krea_modal.py
