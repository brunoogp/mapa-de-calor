import streamlit as st
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

# ── patch do bug torch._classes ---------------------------
import sys, types
_fake = types.ModuleType("torch._classes"); _fake.__path__ = []
sys.modules["torch._classes"] = _fake
# ---------------------------------------------------------


from scipy.ndimage import zoom
from scipy.special import logsumexp
from modelo.deepgaze2e import DeepGazeIIE
from io import BytesIO
import base64
import requests
import os
from dotenv import load_dotenv
import re
import json

load_dotenv()

st.set_page_config(page_title="Mapa de Calor Real", layout="wide")
st.title("Mapa de Atenção Visual")
st.write("Envie uma imagem e visualize onde está a atenção mais provável")

DEVICE = "cpu"

@st.cache_resource
def load_model():
    model = DeepGazeIIE().to(DEVICE)
    return model

model = load_model()
centerbias_template = np.load("centerbias_mit1003.npy")


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def imagem_para_base64(imagem_pil):
    buffered = BytesIO()
    imagem_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def gerar_zonas_dinamicas_com_gemini(imagem_base64, heatmap_base64):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

    prompt = """
Você receberá duas imagens: a original e o mapa de calor visual. Identifique e retorne apenas as coordenadas (em pixels) das seguintes áreas visuais relevantes:
- "elemento de destaque"
- "logo"
- "texto principal"
- "CTA"

O retorno deve ser um JSON no formato:
{
  "elemento de destaque": [x1, y1, x2, y2],
  "logo": [x1, y1, x2, y2],
  "texto principal": [x1, y1, x2, y2],
  "CTA": [x1, y1, x2, y2]
}

Apenas o JSON puro, sem explicações, sem blocos markdown.
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": "image/png", "data": imagem_base64}},
                    {"inlineData": {"mimeType": "image/png", "data": heatmap_base64}},
                ]
            }
        ]
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    raw_output = response.json()["candidates"][0]["content"]["parts"][0]["text"]

    match = re.search(r'{.*}', raw_output, re.DOTALL)
    if not match:
        raise ValueError("❌ O Gemini não retornou um JSON válido.")

    return json.loads(match.group(0))


def gerar_relatorio_com_gemini(imagem_base64, heatmap_base64, estatisticas):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

    prompt = f"""
Você é um especialista em design e publicidade. A imagem a seguir é um criativo publicitário. Junto dela, há um mapa de calor de atenção visual preditiva.

Atenção: você também receberá estatísticas quantitativas sobre o foco visual em áreas específicas da imagem.

Estatísticas de atenção visual:
- Área "elemento de destaque": {estatisticas.get("elemento de destaque", 0)}%
- Área "logo": {estatisticas.get("logo", 0)}%
- Área "texto principal": {estatisticas.get("texto principal", 0)}%
- Área "CTA": {estatisticas.get("CTA", 0)}%

Dê um parecer estratégico com base nesses dados. Siga esta estrutura:

1. Interprete as porcentagens de foco por zona e diga o que funcionou bem.
2. Aponte o que precisa ser melhorado no layout e por quê.
3. Sugira uma nova hierarquia visual ideal para captar mais atenção.
4. Proponha melhorias práticas (cor, contraste, posição dos elementos, etc).
5. Foque em como aumentar a taxa de conversão do CTA e reconhecimento da marca, interpretando se a peça é voltada para branding, vendas ou outro objetivo.
6. Destaque os pontos positivos também. Não invente nem assuma, seja objetivo e analítico.
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": "image/png", "data": imagem_base64}},
                    {"inlineData": {"mimeType": "image/png", "data": heatmap_base64}},
                ]
            }
        ]
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"]


# Tabs da interface
tab1, tab2 = st.tabs(["�� Upload & Análise", "📊 Estatísticas"])

with tab1:
    uploaded_image = st.file_uploader("📷 Envie uma imagem (.jpg, .png)", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        image_resized = image.resize((1024, 1024))
        img_np = np.array(image_resized) / 255.0

        if st.button("🔥 Gerar Mapa de Atenção"):
            st.subheader("Resultado")

            cb = zoom(centerbias_template, (1024 / centerbias_template.shape[0], 1024 / centerbias_template.shape[1]), order=0)
            cb -= logsumexp(cb)

            img_tensor = torch.tensor([img_np.transpose(2, 0, 1)], dtype=torch.float32).to(DEVICE)
            cb_tensor = torch.tensor([cb], dtype=torch.float32).to(DEVICE)

            with torch.no_grad():
                log_density = model(img_tensor, cb_tensor)
                heatmap = torch.exp(log_density)[0, 0].cpu().numpy()

            heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            heatmap_original = zoom(heatmap_normalized, (
                image.size[1] / heatmap_normalized.shape[0],
                image.size[0] / heatmap_normalized.shape[1]
            ), order=1)
            heatmap_colored = plt.get_cmap("jet")(heatmap_original)[..., :3]

            img_original = np.array(image) / 255.0
            blended = 0.6 * img_original + 0.4 * heatmap_colored
            blended = np.clip(blended, 0, 1)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="🖼️ Imagem Original", use_container_width=True)
            with col2:
                st.image(blended, caption="📍 Mapa de Calor Gerado", use_container_width=True)

            img_out = Image.fromarray((blended * 255).astype(np.uint8))
            buffer = BytesIO()
            img_out.save(buffer, format="PNG")
            st.download_button("⬇️ Baixar imagem gerada", buffer.getvalue(), file_name="mapa_calor.png", mime="image/png")

            st.session_state["heatmap"] = heatmap_normalized
            st.session_state["imagem"] = image
            st.session_state["heatmap_img"] = img_out

with tab2:
    if "heatmap" in st.session_state:
        heatmap = st.session_state["heatmap"]
        imagem_base64 = imagem_para_base64(st.session_state["imagem"])
        heatmap_base64 = imagem_para_base64(st.session_state["heatmap_img"])

        zonas = gerar_zonas_dinamicas_com_gemini(imagem_base64, heatmap_base64)

        estatisticas = {}
        total = np.sum(heatmap)

        for nome in ["elemento de destaque", "logo", "texto principal", "CTA"]:
            coords = zonas.get(nome, [])
            if isinstance(coords, list) and len(coords) == 4:
                x1, y1, x2, y2 = coords
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(heatmap.shape[1], x2), min(heatmap.shape[0], y2)
                area = heatmap[y1:y2, x1:x2]
                estatisticas[nome] = round(np.sum(area) / total * 100, 2)
            else:
                estatisticas[nome] = 0.0

        max_y, max_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        percent = heatmap[max_y, max_x] * 100

        st.markdown("### 🔍 Estatísticas Básicas")
        st.markdown(f"- 📍 Ponto de maior atenção: **({max_x}, {max_y})**")
        st.markdown(f"- 🔥 Porcentagem de atenção no ponto máximo: **{percent:.2f}%**")
        for zona, val in estatisticas.items():
            st.markdown(f"- 🎯 Atenção em **{zona}**: {val}%")

        with st.spinner("Gerando relatório com IA (Gemini)..."):
            try:
                relatorio = gerar_relatorio_com_gemini(imagem_base64, heatmap_base64, estatisticas)
            except Exception as e:
                relatorio = f"❌ Erro ao chamar Gemini: {e}"

        st.markdown("---")
        st.markdown("### 🤖 Relatório Estratégico da LLM")
        st.markdown(relatorio)
    else:
        st.info("🔁 Primeiro envie uma imagem na aba anterior.")



