from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from modelo.deepgaze2e import DeepGazeIIE
import numpy as np
import torch
from scipy.ndimage import zoom
from scipy.special import logsumexp
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import base64
import re
import json
import requests
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEVICE = "cpu"

app = FastAPI()

model = None
centerbias_template = None

@app.on_event("startup")
def startup_event():
    global model, centerbias_template
    model = DeepGazeIIE().to(DEVICE)
    centerbias_template = np.load("centerbias_mit1003.npy")

def gerar_heatmap(imagem_pil: Image.Image):
    image_resized = imagem_pil.resize((1024, 1024))
    img_np = np.array(image_resized) / 255.0
    cb = zoom(
        centerbias_template,
        (1024 / centerbias_template.shape[0], 1024 / centerbias_template.shape[1]),
        order=0,
    )
    cb -= logsumexp(cb)

    img_tensor = torch.tensor([img_np.transpose(2, 0, 1)], dtype=torch.float32).to(DEVICE)
    cb_tensor = torch.tensor([cb], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        log_density = model(img_tensor, cb_tensor)
        heatmap = torch.exp(log_density)[0, 0].cpu().numpy()

    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap_orig = zoom(
        heatmap_norm,
        (
            imagem_pil.size[1] / heatmap_norm.shape[0],
            imagem_pil.size[0] / heatmap_norm.shape[1],
        ),
        order=1,
    )
    heatmap_colored = plt.get_cmap("jet")(heatmap_orig)[..., :3]
    blended = 0.6 * (np.array(imagem_pil) / 255.0) + 0.4 * heatmap_colored
    blended = np.clip(blended, 0, 1)

    img_out = Image.fromarray((blended * 255).astype(np.uint8))
    buffer = BytesIO()
    img_out.save(buffer, format="PNG")
    buffer.seek(0)

    total = np.sum(heatmap_norm)
    estatisticas = {}
    max_y, max_x = np.unravel_index(np.argmax(heatmap_norm), heatmap_norm.shape)
    estatisticas["ponto_max"] = [int(max_x), int(max_y)]
    estatisticas["percent_max"] = float(np.max(heatmap_norm) * 100)

    return buffer, estatisticas

@app.post("/api/heatmap/")
async def api_heatmap(file: UploadFile = File(...)):
    content = await file.read()
    imagem_pil = Image.open(BytesIO(content)).convert("RGB")
    buffer, _ = gerar_heatmap(imagem_pil)
    return FileResponse(buffer, media_type="image/png", filename="mapa_calor.png")

@app.post("/api/estatisticas/")
async def api_estatisticas(file: UploadFile = File(...)):
    content = await file.read()
    imagem_pil = Image.open(BytesIO(content)).convert("RGB")
    _, estatisticas = gerar_heatmap(imagem_pil)
    return JSONResponse(content=estatisticas)
