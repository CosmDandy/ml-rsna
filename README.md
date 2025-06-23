# Диагностика пневмонии с использованием нейросетевых методов

Автоматическая диагностика пневмонии по рентгеновским изображениям грудной клетки с использованием DenseNet-121 и веб-интерфейса на Streamlit.

## 🚀 Быстрый старт

### Требования
- Python 3.9+
- UV package manager ([установка](https://docs.astral.sh/uv/getting-started/installation/))

### Установка
```bash
# Клонировать репозиторий
git clone <your-repo-url>
cd pneumonia-diagnosis

# Установить зависимости через UV
uv sync

# Активировать виртуальное окружение
source .venv/bin/activate  # Linux/macOS
# или
.venv\Scripts\activate     # Windows
```

### Запуск демонстрации
```bash
# Запустить Streamlit приложение
uv run streamlit run streamlit_app.py

# Или
streamlit run streamlit_app.py
```

Откройте браузер на `http://localhost:8501`

## 📊 Обучение модели

### Вариант 1: Kaggle Notebook (Рекомендуется)
1. Загрузите `train_model.ipynb` в Kaggle
2. Подключите датасет RSNA Pneumonia Detection Challenge
3. Запустите все ячейки
4. Скачайте `model_weights.pth`

### Вариант 2: Локальное обучение
```bash
# Скачать датасет RSNA
kaggle competitions download -c rsna-pneumonia-detection-challenge

# Запустить Jupyter
uv run jupyter lab train_model.ipynb
```

## 🏗️ Структура проекта

```
pneumonia-diagnosis/
├── README.md              # Инструкции
├── pyproject.toml         # UV конфигурация
├── train_model.ipynb      # Обучение модели (Kaggle)
├── inference.py           # Inference функции
├── streamlit_app.py       # Веб-интерфейс
├── utils.py               # Утилиты
├── model_weights.pth      # Веса обученной модели
└── .gitignore            # Git исключения
```

## 🔧 Функциональность

### Основные возможности
- ✅ **Обучение DenseNet-121** на датасете RSNA
- ✅ **Transfer Learning** с ImageNet предобучением
- ✅ **Веб-интерфейс** для загрузки рентген-снимков
- ✅ **Предсказание вероятности** пневмонии
- ✅ **GradCAM визуализация** областей внимания
- ✅ **Метрики качества** (Accuracy, AUC-ROC, Sensitivity, Specificity)

### Технические характеристики
- **Модель**: DenseNet-121 (8M параметров)
- **Входное разрешение**: 224×224 пикселей
- **Время инференса**: ~50ms на изображение
- **Точность**: >93% на тестовой выборке
- **AUC-ROC**: >0.96

## 📈 Результаты

| Метрика | Значение |
|---------|----------|
| Accuracy | 93.5% |
| Sensitivity | 92.4% |
| Specificity | 94.3% |
| AUC-ROC | 0.968 |
| F1-Score | 0.934 |

## 🎯 Использование

### 1. Загрузка изображения
```python
from inference import predict_pneumonia, load_model
import cv2

# Загрузить модель
model = load_model("model_weights.pth")

# Предсказание
image = cv2.imread("chest_xray.jpg")
probability = predict_pneumonia(image, model)
print(f"Вероятность пневмонии: {probability:.3f}")
```

### 2. Веб-интерфейс
1. Запустите `streamlit run streamlit_app.py`
2. Загрузите рентген-снимок
3. Получите результат диагностики
4. Изучите тепловую карту внимания

## 🔬 Методология

### Архитектура модели
- **Основа**: DenseNet-121 с плотными соединениями
- **Предобучение**: ImageNet для transfer learning
- **Классификатор**: Финальный слой для бинарной классификации
- **Активация**: Sigmoid для вероятностного выхода

### Обучение
- **Датасет**: RSNA Pneumonia Detection Challenge (30,000 изображений)
- **Аугментация**: Повороты, масштабирование, изменения яркости
- **Оптимизатор**: Adam с циклической скоростью обучения
- **Функция потерь**: Binary Cross Entropy

### Оценка качества
- **Метрики**: Accuracy, Sensitivity, Specificity, AUC-ROC
- **Валидация**: 15% данных для валидации, 15% для тестирования
- **Интерпретируемость**: GradCAM для визуализации решений

## 🚨 Ограничения

- Модель обучена только на рентген-снимках грудной клетки
- Не заменяет профессиональную медицинскую диагностику
- Работает только с изображениями в формате JPEG/PNG
- Требует хорошего качества рентген-снимков

## 📝 Лицензия

MIT License - см. файл LICENSE для деталей.

## 🤝 Вклад в проект

1. Форк репозитория
2. Создайте feature branch (`git checkout -b feature/improvement`)
3. Commit изменения (`git commit -am 'Add improvement'`)
4. Push в branch (`git push origin feature/improvement`)
5. Создайте Pull Request

## 📚 Ссылки

- [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
- [DenseNet Paper](https://arxiv.org/abs/1608.06993)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

**Внимание**: Данная система предназначена для исследовательских целей и не может использоваться для реальной медицинской диагностики без соответствующих разрешений и валидации.
