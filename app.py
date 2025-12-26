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

st.title("UNet Semantic Segmentation")

# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
weights_path = "models/unet_forest_weights_finetuned.pth"

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

transform = A.Compose([
    A.Resize(256, 256),  # размер как при обучении
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# -----------------------------
def predict_image(img_pil):
    img_np = np.array(img_pil.convert("RGB"))
    augmented = transform(image=img_np)
    img_tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))
    pred_mask = (pred > 0.5).float().cpu().squeeze(0).squeeze(0).numpy()
    pred_mask_resized = cv2.resize(pred_mask, (img_np.shape[1], img_np.shape[0]))
    return img_np, pred_mask_resized

# -----------------------------
st.subheader("Загрузка изображений")

# Загрузка с компьютера
uploaded_files = st.file_uploader(
    "Выберите изображения", type=["jpg","jpeg","png"], accept_multiple_files=True
)

# Загрузка по URL
urls_text = st.text_area("Или вставьте URL изображений (по одному на строку)")

images = []

# Файлы с компьютера
if uploaded_files:
    for file in uploaded_files:
        images.append(Image.open(file))

# Файлы по URL
if urls_text:
    urls = urls_text.split("\n")
    for url in urls:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers)
            img_pil = Image.open(BytesIO(response.content))
            images.append(img_pil)
        except UnidentifiedImageError:
            st.warning(f"Не удалось загрузить изображение по URL: {url}")

# -----------------------------
# Предсказание и визуализация
for idx, img_pil in enumerate(images):
    img_np, mask_np = predict_image(img_pil)

    st.markdown(f"### Изображение {idx+1}")
    fig, ax = plt.subplots(1,2, figsize=(12,6))
    ax[0].imshow(img_np)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(img_np)
    ax[1].imshow(mask_np, cmap='Reds', alpha=0.5)
    ax[1].set_title("Predicted Mask")
    ax[1].axis("off")
    st.pyplot(fig)

# -----------------------------
st.subheader("Информация о модели")
st.markdown("""
- Архитектура: UNet (ResNet34)  
- Входные каналы: 3  
- Выходные каналы: 1  
- Метрика: DiceLoss  
- Эпохи обучения: 10  
- Размер выборки: 500 изображений  
- Метрики качества: Dice, IoU
""")
