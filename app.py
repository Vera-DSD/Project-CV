import streamlit as st

# =============================
# Конфигурация страницы
# =============================
st.set_page_config(
    page_title="CV Project",
    layout="wide"
)

# =============================
# Баннер / заголовок
# =============================
st.markdown(
    """
    <div style="background-color:#4CAF50;padding:30px;border-radius:10px">
        <h1 style="color:white;text-align:center;"> CV Project: Computer Vision</h1>
        <p style="color:white;text-align:center;font-size:18px;">
        Мультизадачное приложение для сегментации и детекции объектов
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# =============================
# Описание проекта
# =============================
st.markdown(
    """
    Добро пожаловать! Это проект демонстрирует возможности компьютерного зрения:
    """
)

# =============================
# Карточки для модулей
# =============================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div style="
            background-color:#E0F7FA;
            padding:20px;
            border-radius:10px;
            text-align:center;
            height:200px;  /* Увеличиваем высоту банера */
            display:flex;
            flex-direction:column;
            justify-content:center;
        ">
            <h3>🌲 Семантическая сегментация</h3>
            <p>UNet для сегментации лесных снимков.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div style="
            background-color:#FFF3E0;
            padding:20px;
            border-radius:10px;
            text-align:center;
            height:200px;  /* Увеличиваем высоту банера */
            display:flex;
            flex-direction:column;
            justify-content:center;
        ">
            <h3>😃 Детекция лиц</h3>
            <p>YOLO для обнаружения лиц на изображениях.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div style="
            background-color:#E8F5E9;
            padding:20px;
            border-radius:10px;
            text-align:center;
            height:200px;  /* Увеличиваем высоту банера */
            display:flex;
            flex-direction:column;
            justify-content:center;
        ">
            <h3>⚡ Детекция объектов</h3>
            <p>YOLO для обнаружения объектов, например, ветрогенераторов.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# =============================
# Дополнительная информация
# =============================
st.markdown(
    """
    **Навигация:**  
    Используйте боковое меню слева, чтобы перейти к интересующему модулю.

    """)