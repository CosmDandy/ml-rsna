"""
Streamlit веб-приложение для диагностики пневмонии.
Обеспечивает интуитивный интерфейс для загрузки рентген-снимков и получения диагностики.
"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import plotly.graph_objects as go
import pydicom

# Импорты наших модулей
from inference import (
    load_model,
    predict_pneumonia,
    generate_gradcam_visualization,
    create_diagnosis_report,
)

# Настройка страницы
st.set_page_config(
    page_title="Диагностика пневмонии",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Кэширование загрузки модели
@st.cache_resource
def load_cached_model():
    """Загрузка и кэширование модели."""
    try:
        model = load_model("./model_weights.pth")
        return model, True
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None, False


# Функция для создания HTML с результатами
def create_result_html(probability, confidence, classification, risk_level):
    """Создание HTML блока с результатами."""

    # Цвета для разных уровней риска
    color_map = {
        "Very Low": "#28a745",  # Зеленый
        "Low": "#6fb83f",  # Светло-зеленый
        "Moderate": "#ffc107",  # Желтый
        "High": "#fd7e14",  # Оранжевый
        "Very High": "#dc3545",  # Красный
    }

    risk_color = color_map.get(risk_level, "#6c757d")

    # Заголовок приложения
    html = f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    ">
        <h2 style="margin-bottom: 1rem; font-size: 1.8rem;">📊 Результат диагностики</h2>
        <div style="
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            backdrop-filter: blur(10px);
        ">
            <h3 style="margin-bottom: 0.5rem;">Классификация</h3>
            <p style="font-size: 1.5rem; font-weight: bold; margin: 0;">
                {classification}
            </p>
        </div>
        <div style="display: flex; justify-content: space-around; margin: 1rem 0;">
            <div style="text-align: center;">
                <h4>Вероятность пневмонии</h4>
                <p style="font-size: 2rem; font-weight: bold; color: #ffd700;">
                    {probability:.1%}
                </p>
            </div>
            <div style="text-align: center;">
                <h4>Уверенность модели</h4>
                <p style="font-size: 2rem; font-weight: bold; color: #87ceeb;">
                    {confidence:.1%}
                </p>
            </div>
        </div>
        <div style="
            background: {risk_color};
            border-radius: 10px;
            padding: 1rem;
            margin-top: 1rem;
        ">
            <h4>Уровень риска: {risk_level}</h4>
        </div>
    </div>
    """

    return html


# Функция для создания Plotly графика вероятности
def create_probability_gauge(probability):
    """Создание круглого индикатора вероятности."""

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Вероятность пневмонии (%)"},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 25], "color": "lightgreen"},
                    {"range": [25, 50], "color": "yellow"},
                    {"range": [50, 75], "color": "orange"},
                    {"range": [75, 100], "color": "red"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        )
    )

    fig.update_layout(
        height=300, font=dict(size=16), margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


# Главная функция приложения
def main():
    """Основная функция Streamlit приложения."""

    # Заголовок приложения
    st.markdown(
        """
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            margin-bottom: 0.5rem;
        ">
            🫁 Диагностика пневмонии
        </h1>
        <p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
            Автоматический анализ рентген-снимков с использованием DenseNet-121
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Боковая панель с информацией
    with st.sidebar:
        st.markdown("## 📋 Информация о модели")

        model_info = {
            "Архитектура": "DenseNet-121",
            "Входной размер": "224×224 пикселей",
            "Точность": ">93%",
            "AUC-ROC": ">0.96",
            "Время обработки": "~50ms",
        }

        for key, value in model_info.items():
            st.markdown(f"**{key}:** {value}")

        st.markdown("---")
        st.markdown("## ⚠️ Важные ограничения")
        st.markdown(
            """
        - Только для исследовательских целей
        - Не заменяет профессиональную диагностику
        - Требует рентген-снимков хорошего качества
        - Работает только с изображениями грудной клетки
        """
        )

        st.markdown("---")
        st.markdown("## 📊 Поддерживаемые форматы")
        st.markdown("- JPEG, PNG, DICOM")
        st.markdown("- Максимальный размер: 10MB")

    # Загрузка модели
    with st.spinner("Загрузка модели..."):
        model, model_loaded = load_cached_model()

    if not model_loaded:
        st.error(
            "❌ Не удалось загрузить модель. Проверьте наличие файла model_weights.pth"
        )
        st.stop()

    st.success("✅ Модель загружена успешно!")

    # СТРОКА 1: Блок загрузки изображений
    st.markdown("### 📤 Загрузка изображения")

    upload_col1, upload_col2 = st.columns([1, 1])

    uploaded_image = None

    with upload_col1:
        st.markdown("#### Выбор файла")

        uploaded_file = st.file_uploader(
            "Выберите рентген-снимок",
            type=["jpg", "jpeg", "png", "dcm"],
            accept_multiple_files=False,
        )

        if uploaded_file is not None:
            # Проверка размера файла
            if uploaded_file.size > 10 * 1024 * 1024:  # 10MB
                st.error(
                    "Размер файла превышает 10MB. Пожалуйста, загрузите меньший файл."
                )
                st.stop()

            # Загрузка изображения
            try:
                if uploaded_file.type == "application/dicom":
                    dicom_data = pydicom.dcmread(uploaded_file)
                    image_array = dicom_data.pixel_array
                    # Нормализация
                    image_array = (
                        (image_array - image_array.min())
                        / (image_array.max() - image_array.min())
                        * 255
                    ).astype(np.uint8)
                    uploaded_image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
                else:
                    # Обычные форматы изображений
                    uploaded_image = np.array(Image.open(uploaded_file).convert("RGB"))

            except Exception as e:
                st.error(f"Ошибка загрузки изображения: {e}")
                st.stop()

        # Информация об изображении (показываем только если изображение загружено)
        if uploaded_image is not None:
            st.markdown("#### Информация о файле")
            file_size_mb = uploaded_file.size / (1024 * 1024)
            st.markdown(f"**Имя файла:** {uploaded_file.name} ({file_size_mb:.2f} MB)")
            st.markdown(
                f"**Размер:** {uploaded_image.shape[0]}×{uploaded_image.shape[1]} пикселей"
            )
            st.markdown(
                f"**Тип:** {'Цветное' if len(uploaded_image.shape) == 3 else 'Черно-белое'}"
            )

    with upload_col2:
        if uploaded_image is not None:
            st.markdown("#### Загруженное изображение")
            st.image(
                uploaded_image,
                caption="Рентген-снимок грудной клетки",
                use_container_width=True,
            )
        else:
            st.markdown("#### Предварительный просмотр")
            st.info("Загрузите изображение для предварительного просмотра")

    # СТРОКА 2: Результаты анализа
    if uploaded_image is not None:
        st.markdown("---")
        st.markdown("### 🔬 Результаты анализа")

        # Кнопка анализа на всю ширину
        if st.button("Выполнить диагностику", type="primary", use_container_width=True):
            with st.spinner("Анализ изображения..."):
                try:
                    # Получение предсказания
                    probability, confidence = predict_pneumonia(
                        uploaded_image, model, return_confidence=True
                    )

                    # Определение классификации
                    classification = (
                        "Пневмония обнаружена" if probability >= 0.5 else "Норма"
                    )

                    # Определение уровня риска
                    if probability >= 0.8:
                        risk_level = "Very High"
                    elif probability >= 0.6:
                        risk_level = "High"
                    elif probability >= 0.4:
                        risk_level = "Moderate"
                    elif probability >= 0.2:
                        risk_level = "Low"
                    else:
                        risk_level = "Very Low"

                    # Создание отчета
                    report = create_diagnosis_report(uploaded_image, model)

                    # Разделяем результаты на две колонки
                    result_col1, result_col2 = st.columns([1, 1])

                    with result_col1:
                        # Отображение основных результатов
                        st.markdown(
                            create_result_html(
                                probability, confidence, classification, risk_level
                            ),
                            unsafe_allow_html=True,
                        )

                    with result_col2:
                        # Круговой индикатор
                        st.plotly_chart(
                            create_probability_gauge(probability),
                            use_container_width=True,
                        )

                        # Отображение рекомендаций
                        st.markdown("#### Клинические рекомендации")
                        for i, recommendation in enumerate(
                            report["recommendations"], 1
                        ):
                            st.markdown(f"{i}. {recommendation}")

                except Exception as e:
                    st.error(f"Ошибка при анализе изображения: {e}")

    # Секция с Grad-CAM визуализацией
    if uploaded_image is not None:
        st.markdown("---")
        st.markdown("### 🔍 Grad-CAM визуализация")

        if st.button(
            "Создать тепловую карту", type="primary", use_container_width=True
        ):
            with st.spinner("Генерация Grad-CAM..."):
                try:
                    original, overlay, prob = generate_gradcam_visualization(
                        uploaded_image, model
                    )

                    col3, col4 = st.columns([1, 1])

                    with col3:
                        st.markdown("#### Оригинальное изображение")
                        st.image(original, use_container_width=True)

                    with col4:
                        st.markdown("#### Области внимания модели")
                        st.image(overlay, use_container_width=True)
                        st.caption(
                            f"Красные области указывают на признаки пневмонии (вероятность: {prob:.3f})"
                        )

                except Exception as e:
                    st.error(f"Ошибка создания Grad-CAM: {e}")

    # Секция с пакетным анализом
    st.markdown("---")
    st.markdown("### 📊 Пакетный анализ")

    uploaded_files = st.file_uploader(
        "Загрузите несколько изображений для пакетного анализа",
        type=["jpg", "jpeg", "png", "dcm"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.markdown(f"Загружено изображений: {len(uploaded_files)}")

        if st.button("Анализировать все изображения"):
            progress_bar = st.progress(0)
            results = []

            for i, file in enumerate(uploaded_files):
                try:
                    if file.type == "application/dicom":
                        dicom_data = pydicom.dcmread(file)
                        image_array = dicom_data.pixel_array
                        # Нормализация
                        image_array = (
                            (image_array - image_array.min())
                            / (image_array.max() - image_array.min())
                            * 255
                        ).astype(np.uint8)
                        image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
                    else:
                        # Обычные форматы изображений
                        image = np.array(Image.open(file).convert("RGB"))

                    probability, confidence = predict_pneumonia(
                        image, model, return_confidence=True
                    )

                    results.append(
                        {
                            "Файл": file.name,
                            "Вероятность пневмонии": f"{probability:.3f}",
                            "Уверенность": f"{confidence:.3f}",
                            "Классификация": (
                                "Пневмония" if probability >= 0.5 else "Норма"
                            ),
                        }
                    )

                    progress_bar.progress((i + 1) / len(uploaded_files))

                except Exception as e:
                    st.error(f"Ошибка обработки {file.name}: {e}")

            # Отображение результатов в таблице
            if results:
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)

                # Статистика
                pneumonia_count = sum(
                    1 for r in results if r["Классификация"] == "Пневмония"
                )
                st.markdown(
                    f"**Статистика:** {pneumonia_count} из {len(results)} изображений классифицированы как пневмония"
                )

    # Футер
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🎓 Выпускная квалификационная работа по диагностике пневмонии</p>
        <p>Модель основана на DenseNet-121 и обучена на датасете RSNA Pneumonia Detection Challenge</p>
        <p><small>⚠️ Только для исследовательских и образовательных целей</small></p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
