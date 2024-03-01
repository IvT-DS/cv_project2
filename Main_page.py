import streamlit as st
from PIL import Image

st.sidebar.markdown("## Используй навигацию между страницами выше ⬆️")
st.sidebar.markdown("# Главная страница -->")



image = Image.open("pages/CV.jpg")
st.image(image, use_column_width=True)

st.markdown("## Проект на тему <<Компьютерное зрение>>")

st.markdown("### Фаза 2 / неделя 2.")
st.markdown("#### Команда проекта 👨🏻‍💻")
st.write(
    """
    <style>
    .my-text {
        line-height: 0.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.write('<p class="my-text">1. Иван Терещенко</p>', unsafe_allow_html=True)
st.write('<p class="my-text">2. Артем Маслов</p>', unsafe_allow_html=True)

st.markdown("#### Задачи 📜")

st.write(
    """
    <style>
    .my-text {
        line-height: 0.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.write(
    '<p class="my-text">1. Разработать multipage-приложение с использованием streamlit.</p>',
    unsafe_allow_html=True,
)
st.write(
    '<p class="my-text">2. Выбрать модель YOLO. Обучить для детекции судов на изображениях аэросъемки.</p>',
    unsafe_allow_html=True,
)

st.write(
    '<p class="my-text">3. Обучить автоэнкодер для очистки документов от шумов.</p>',
    unsafe_allow_html=True,
)

st.markdown("#### Требования к проекту ✏️")

st.write(
    """
    <style>
    .my-text {
        line-height: 0.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
- Все страницы должны поддерживать загрузку пользователем сразу нескольких файлов.
- Страница с детекцией объектов должна поддерживать подгрузку файла по прямой ссылке.
- Все страницы должны иметь раздел с информацией о моделях, качестве и процессе обучения:
    - Число эпох обучения.
    - Объем выборок.
    - Метрики (для детекции mAP, график PR кривой, confusion matrix; для модели очищения от шума – RMSE).
"""
)


