"""
Inference модуль для диагностики пневмонии.
Включает загрузку модели, предсказания и визуализацию результатов.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Tuple, Optional, Union
import warnings

from utils import preprocess_image, denormalize_tensor, IMAGENET_MEAN, IMAGENET_STD

warnings.filterwarnings("ignore")


class PneumoniaClassifier(nn.Module):
    """
    Классификатор пневмонии на основе DenseNet-121.
    """

    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super(PneumoniaClassifier, self).__init__()

        # Загружаем предобученную DenseNet-121
        self.densenet = models.densenet121(pretrained=pretrained)

        # Заменяем классификатор
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes),
            nn.Sigmoid(),  # Для бинарной классификации
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.densenet(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Извлечение признаков из последнего сверточного слоя."""
        features = self.densenet.features(x)
        return F.adaptive_avg_pool2d(features, (1, 1))


class GradCAM:
    """
    Реализация Gradient-weighted Class Activation Mapping (Grad-CAM).
    """

    def __init__(self, model: nn.Module, target_layer: str = "features"):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Регистрируем хуки
        self._register_hooks()

    def _register_hooks(self):
        """Регистрация хуков для извлечения градиентов и активаций."""

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Получаем целевой слой
        if hasattr(self.model, "densenet"):
            target = getattr(self.model.densenet, self.target_layer)
        else:
            target = getattr(self.model, self.target_layer)

        target.register_forward_hook(forward_hook)
        target.register_backward_hook(backward_hook)

    def generate_cam(
        self, input_tensor: torch.Tensor, class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Генерация Grad-CAM карты.

        Args:
            input_tensor: Входной тензор
            class_idx: Индекс класса (None для максимального класса)

        Returns:
            Grad-CAM карта
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = 0  # Для бинарной классификации

        # Backward pass
        self.model.zero_grad()
        class_score = output[:, class_idx] if output.dim() > 1 else output
        class_score.backward()

        # Получаем градиенты и активации
        gradients = self.gradients
        activations = self.activations

        # Вычисляем веса
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)

        # Генерируем Grad-CAM
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Нормализуем
        cam = cam.squeeze()
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


def load_model(model_path: str, device: Optional[str] = None) -> PneumoniaClassifier:
    """
    Загрузка обученной модели.

    Args:
        model_path: Путь к файлу модели
        device: Устройство для вычислений

    Returns:
        Загруженная модель
    """
    # if device is None:
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    # Проверка доступности MPS
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Используется GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Используется GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Используется CPU")

    # Создаем модель
    model = PneumoniaClassifier(num_classes=1, pretrained=False)

    # Загружаем веса
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)

            # Если это checkpoint с дополнительной информацией
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"Модель загружена из checkpoint: {model_path}")
                if "metrics" in checkpoint:
                    print(f"Метрики модели: {checkpoint['metrics']}")
            else:
                # Если это только state_dict
                model.load_state_dict(checkpoint)
                print(f"Модель загружена: {model_path}")

        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            print("Использую предобученную модель без дообучения")
            model = PneumoniaClassifier(num_classes=1, pretrained=True)
    else:
        print(f"Файл модели не найден: {model_path}")
        print("Использую предобученную модель без дообучения")
        model = PneumoniaClassifier(num_classes=1, pretrained=True)

    model.to(device)
    model.eval()

    return model


def predict_pneumonia(
    image: Union[np.ndarray, str, Image.Image],
    model: PneumoniaClassifier,
    device: Optional[str] = None,
    return_confidence: bool = True,
) -> Union[float, Tuple[float, float]]:
    """
    Предсказание вероятности пневмонии.

    Args:
        image: Входное изображение
        model: Обученная модель
        device: Устройство для вычислений
        return_confidence: Возвращать ли уверенность

    Returns:
        Вероятность пневмонии (и уверенность, если запрошена)
    """
    # if device is None:
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Используется GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Используется GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Используется CPU")

    # Предобработка изображения
    input_tensor = preprocess_image(image)
    input_tensor = input_tensor.to(device)

    # Предсказание
    with torch.no_grad():
        output = model(input_tensor)
        probability = output.item()

    # Вычисляем уверенность как расстояние от 0.5
    confidence = abs(probability - 0.5) * 2

    if return_confidence:
        return probability, confidence
    else:
        return probability


def generate_gradcam_visualization(
    image: Union[np.ndarray, str, Image.Image],
    model: PneumoniaClassifier,
    device: Optional[str] = None,
    alpha: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Генерация Grad-CAM визуализации.

    Args:
        image: Входное изображение
        model: Обученная модель
        device: Устройство для вычислений
        alpha: Прозрачность наложения

    Returns:
        Tuple(оригинальное изображение, визуализация с наложением, вероятность)
    """
    # if device is None:
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Используется GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Используется GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Используется CPU")

    # Загружаем изображение как numpy array для визуализации
    if isinstance(image, str):
        original_image = cv2.imread(image)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        original_image = image.copy()
    elif isinstance(image, Image.Image):
        original_image = np.array(image.convert("RGB"))

    # Изменяем размер для отображения
    original_image = cv2.resize(original_image, (224, 224))

    # Предобработка для модели
    input_tensor = preprocess_image(image)
    input_tensor = input_tensor.to(device)

    # Получаем предсказание
    probability = predict_pneumonia(image, model, device, return_confidence=False)

    # Генерируем Grad-CAM
    gradcam = GradCAM(model)
    cam = gradcam.generate_cam(input_tensor)

    # Создаем тепловую карту
    heatmap = cm.jet(cam)[:, :, :3]  # Убираем альфа канал
    heatmap = (heatmap * 255).astype(np.uint8)

    # Накладываем тепловую карту на изображение
    overlay = original_image * (1 - alpha) + heatmap * alpha
    overlay = overlay.astype(np.uint8)

    return original_image, overlay, probability


def analyze_image_batch(
    image_paths: list,
    model: PneumoniaClassifier,
    device: Optional[str] = None,
    threshold: float = 0.5,
) -> dict:
    """
    Анализ батча изображений.

    Args:
        image_paths: Список путей к изображениям
        model: Обученная модель
        device: Устройство для вычислений
        threshold: Порог классификации

    Returns:
        Словарь с результатами анализа
    """
    # if device is None:
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Используется GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Используется GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Используется CPU")

    results = {
        "predictions": [],
        "confidences": [],
        "classifications": [],
        "paths": image_paths,
    }

    for image_path in image_paths:
        try:
            probability, confidence = predict_pneumonia(
                image_path, model, device, return_confidence=True
            )
            classification = "Pneumonia" if probability >= threshold else "Normal"

            results["predictions"].append(probability)
            results["confidences"].append(confidence)
            results["classifications"].append(classification)

        except Exception as e:
            print(f"Ошибка обработки {image_path}: {e}")
            results["predictions"].append(None)
            results["confidences"].append(None)
            results["classifications"].append("Error")

    return results


def create_diagnosis_report(
    image: Union[np.ndarray, str, Image.Image],
    model: PneumoniaClassifier,
    patient_info: Optional[dict] = None,
    device: Optional[str] = None,
) -> dict:
    """
    Создание диагностического отчета.

    Args:
        image: Изображение для анализа
        model: Обученная модель
        patient_info: Информация о пациенте
        device: Устройство для вычислений

    Returns:
        Диагностический отчет
    """
    # if device is None:
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Используется GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Используется GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Используется CPU")

    # Получаем предсказание
    probability, confidence = predict_pneumonia(
        image, model, device, return_confidence=True
    )

    # Определяем классификацию
    classification = "Pneumonia" if probability >= 0.5 else "Normal"

    # Определяем уровень риска
    if probability >= 0.8:
        risk_level = "High"
    elif probability >= 0.6:
        risk_level = "Moderate"
    elif probability >= 0.4:
        risk_level = "Low"
    else:
        risk_level = "Very Low"

    # Создаем отчет
    report = {
        "diagnosis": {
            "classification": classification,
            "probability": round(probability, 3),
            "confidence": round(confidence, 3),
            "risk_level": risk_level,
        },
        "recommendations": [],
        "patient_info": patient_info or {},
    }

    # Добавляем рекомендации
    if probability >= 0.7:
        report["recommendations"].extend(
            [
                "Немедленная консультация врача",
                "Дополнительные исследования (КТ, анализы)",
                "Рассмотреть начало антибиотикотерапии",
            ]
        )
    elif probability >= 0.5:
        report["recommendations"].extend(
            [
                "Консультация врача в ближайшее время",
                "Динамическое наблюдение",
                "Повторная рентгенография через 24-48 часов",
            ]
        )
    else:
        report["recommendations"].extend(
            [
                "Динамическое наблюдение",
                "При ухудшении симптомов - повторное обследование",
            ]
        )

    return report


def save_visualization(
    original_image: np.ndarray,
    overlay_image: np.ndarray,
    probability: float,
    save_path: str,
    title: Optional[str] = None,
):
    """
    Сохранение визуализации результатов.

    Args:
        original_image: Оригинальное изображение
        overlay_image: Изображение с наложением
        probability: Вероятность пневмонии
        save_path: Путь для сохранения
        title: Заголовок
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Оригинальное изображение
    axes[0].imshow(original_image)
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

    # Изображение с Grad-CAM
    axes[1].imshow(overlay_image)
    axes[1].set_title(f"Grad-CAM (Pneumonia: {probability:.3f})")
    axes[1].axis("off")

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# Главная функция для тестирования
def main():
    """Тестирование inference модуля."""
    print("Тестирование inference модуля...")

    # Проверяем доступность CUDA
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Используемое устройство: {device}")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Используется GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Используется GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Используется CPU")

    # Загружаем модель (будет использована предобученная, если веса не найдены)
    model = load_model("./model_weights.pth", device)
    print("Модель загружена успешно")

    # Тестируем на случайном изображении
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    # Предсказание
    probability, confidence = predict_pneumonia(
        test_image, model, device, return_confidence=True
    )
    print(f"Вероятность пневмонии: {probability:.3f}")
    print(f"Уверенность: {confidence:.3f}")

    # Grad-CAM визуализация
    try:
        original, overlay, prob = generate_gradcam_visualization(
            test_image, model, device
        )
        print("Grad-CAM визуализация создана успешно")
    except Exception as e:
        print(f"Ошибка создания Grad-CAM: {e}")

    # Диагностический отчет
    report = create_diagnosis_report(test_image, model, device=device)
    print("Диагностический отчет:")
    print(f"  Классификация: {report['diagnosis']['classification']}")
    print(f"  Уровень риска: {report['diagnosis']['risk_level']}")
    print(f"  Рекомендации: {len(report['recommendations'])} пунктов")

    print("Все тесты пройдены успешно!")


if __name__ == "__main__":
    main()
