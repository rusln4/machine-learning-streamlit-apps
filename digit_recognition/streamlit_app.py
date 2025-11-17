import os
import io
import requests
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Распознавание цифр", page_icon="")

st.title("Распознавание цифр")
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8001/predict")
stroke_width = st.slider("Толщина линии", 6, 40, 20)
if "canvas_nonce" not in st.session_state:
    st.session_state["canvas_nonce"] = 0
col_del, col_rec = st.columns(2)
with col_del:
    delete_clicked = st.button("Удалить")
with col_rec:
    recognize_clicked = st.button("Распознать")
if delete_clicked:
    st.session_state["canvas_nonce"] += 1

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=stroke_width,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    display_toolbar=False,
    initial_drawing={"version": "4.4.0"},
    key=f"canvas-{st.session_state['canvas_nonce']}",
)

if recognize_clicked and canvas_result.image_data is not None:
    img_data = canvas_result.image_data.astype(np.uint8)
    img = Image.fromarray(img_data).convert("L")
    img_small = ImageOps.fit(img, (28, 28), method=Image.LANCZOS)
    buf = io.BytesIO()
    img_small.save(buf, format="PNG")
    buf.seek(0)
    try:
        r = requests.post(API_URL, files={"file": ("digit.png", buf, "image/png")}, timeout=10)
        if not r.ok:
            st.error(f"Ошибка API: {r.status_code}")
        else:
            data = r.json()
            if "prediction" in data and "probabilities" in data:
                st.success(f"Предсказание: {data['prediction']}")
                st.bar_chart(data["probabilities"])
            else:
                st.error(f"Некорректный ответ API: {data}")
    except Exception as e:
        st.error(f"Не удалось обратиться к API: {e}")