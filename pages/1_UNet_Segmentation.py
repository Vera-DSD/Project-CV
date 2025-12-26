import streamlit as st
import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import requests
import cv2
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -----------------------------
st.set_page_config(page_title="UNet Forest Segmentation", layout="wide")
st.title("🌲 UNet Forest Segmentation")

device = "cuda" if torch.cuda.is_available() else "cpu"
weights_path = "unet_forest_weights_final.pth"

# -----------------------------
@st.cache_resource
def load_model(weights_path):
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model(weights_path)

# -----------------------------
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])

# -----------------------------
def predict_image(img_pil):
    img_np = np.array(img_pil.convert("RGB"))

    augmented = transform(image=img_np)
    img_tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))

    mask = (pred > 0.5).float().cpu().squeeze().numpy()
    mask = cv2.resize(mask, (img_np.shape[1], img_np.shape[0]))

    return img_np, mask

# =============================
# ЗАГРУЗКА ИЗОБРАЖЕНИЙ
# =============================
st.header("📤 Загрузка изображений")

uploaded_files = st.file_uploader(
    "Загрузите изображения с компьютера",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

urls_text = st.text_area(
    "Или вставьте URL изображений (по одному на строку)",
    placeholder="https://example.com/image1.jpg\nhttps://example.com/image2.jpg"
)

load_urls_btn = st.button("🌐 Загрузить изображения по URL")

images = []

# --- Локальные файлы ---
if uploaded_files:
    for file in uploaded_files:
        images.append(Image.open(file))

# --- URL (ТОЛЬКО ПО КНОПКЕ) ---
if load_urls_btn and urls_text.strip():
    urls = [u.strip() for u in urls_text.split("\n") if u.strip()]

    for url in urls:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            img = Image.open(BytesIO(response.content)).convert("RGB")
            images.append(img)

        except (UnidentifiedImageError, requests.RequestException):
            st.warning(f"❌ Не удалось загрузить изображение: {url}")

# =============================
# ВИЗУАЛИЗАЦИЯ
# =============================
if images:
    st.header("🧠 Результаты сегментации")

    for idx, img_pil in enumerate(images):
        img_np, mask_np = predict_image(img_pil)

        # Накладываем маску вручную
        overlay = img_np.copy()
        red_mask = np.zeros_like(overlay)
        red_mask[..., 0] = 255
        overlay = np.where(mask_np[..., None] > 0, 
                            0.6 * red_mask + 0.4 * overlay, 
                            overlay).astype(np.uint8)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Оригинал #{idx+1}**")
            st.image(img_np, width=500)

        with col2:
            st.markdown(f"**Сегментация #{idx+1}**")
            st.image(overlay, width=500)

# =============================
# ИНФОРМАЦИЯ О МОДЕЛИ
# =============================
st.header("ℹ️ Информация о модели")
st.markdown("""
**Архитектура:** UNet (ResNet34)  
**Тип задачи:** Семантическая сегментация   

**Обучение:**
- Эпохи: 30  
- Размер изображений: 256×256  
- Датасет: Forest Aerial Images  

**Метрики:**
- Train : Loss: 0.1404 | Dice: 0.8596 | IoU: 0.7587 | Acc: 0.8204 | AP50: 0.8118 | AP75: 0.8120 
- Valid : Loss: 0.1407 | Dice: 0.8593 | IoU: 0.7587 | Acc: 0.8259 | AP50: 0.8167 | AP75: 0.8169 

""")
