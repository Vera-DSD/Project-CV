# import streamlit as st
# import torch
# from PIL import Image
# import cv2
# import numpy as np

# st.title("YOLO Face Detection")

# device = "cuda" if torch.cuda.is_available() else "cpu"
# weights_path = "models/yolo_face.pt"

# @st.cache_resource
# def load_yolo_model(weights_path):
#     model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)
#     model.to(device)
#     return model

# model = load_yolo_model(weights_path)

# uploaded_files = st.file_uploader("Загрузите изображения", type=["jpg","png"], accept_multiple_files=True)
# urls_text = st.text_area("Или вставьте URL изображений (по одному на строку)")

# images = []
# if uploaded_files:
#     for file in uploaded_files:
#         images.append(Image.open(file))

# if urls_text:
#     import requests
#     from io import BytesIO
#     urls = urls_text.split("\n")
#     for url in urls:
#         try:
#             response = requests.get(url)
#             img_pil = Image.open(BytesIO(response.content))
#             images.append(img_pil)
#         except:
#             st.warning(f"Не удалось загрузить изображение по URL: {url}")

# for idx, img_pil in enumerate(images):
#     results = model(np.array(img_pil))
#     results.render()  # накладываем предсказания
#     img_rendered = Image.fromarray(results.ims[0])
#     st.markdown(f"### Изображение {idx+1}")
#     st.image(img_rendered)