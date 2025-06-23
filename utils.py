"""
Утилиты для проекта диагностики пневмонии.
Включает функции предобработки изображений, трансформации и вспомогательные функции.
"""

import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, Union
import torchvision.transforms as transforms
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    confusion_matrix, 
    classification_report,
    roc_curve
)


# Константы для нормализации ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Размер входного изображения для модели
INPUT_SIZE = 224


def preprocess_image(
    image: Union[np.ndarray, str, Image.Image], 
    target_size: int = INPUT_SIZE,
    normalize: bool = True
) -> torch.Tensor:
    """
    Предобработка изображения для inference.
    
    Args:
        image: Входное изображение (numpy array, путь к файлу или PIL Image)
        target_size: Целевой размер изображения
        normalize: Применять ли нормализацию ImageNet
    
    Returns:
        Preprocessed tensor готовый для модели
    """
    # Загрузка изображения в зависимости от типа входа
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Файл не найден: {image}")
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:
            img = image
        else:
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif isinstance(image, Image.Image):
        img = np.array(image.convert('RGB'))
    else:
        raise ValueError("Неподдерживаемый тип изображения")
    
    # Базовые трансформации
    transform_list = [
        transforms.ToPILImage(),
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
    ]
    
    # Добавляем нормализацию если требуется
    if normalize:
        transform_list.append(
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        )
    
    transform = transforms.Compose(transform_list)
    
    # Применяем трансформации
    tensor = transform(img)
    
    # Добавляем batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor


def get_train_transforms(
    input_size: int = INPUT_SIZE,
    augment: bool = True
) -> transforms.Compose:
    """
    Трансформации для обучения с аугментацией.
    
    Args:
        input_size: Размер входного изображения
        augment: Применять ли аугментацию
    
    Returns:
        Composed transforms для обучения
    """
    transform_list = [
        transforms.Resize((input_size, input_size)),
    ]
    
    if augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
        ])
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return transforms.Compose(transform_list)


def get_val_transforms(input_size: int = INPUT_SIZE) -> transforms.Compose:
    """
    Трансформации для валидации без аугментации.
    
    Args:
        input_size: Размер входного изображения
    
    Returns:
        Composed transforms для валидации
    """
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: list = IMAGENET_MEAN,
    std: list = IMAGENET_STD
) -> torch.Tensor:
    """
    Денормализация тензора для визуализации.
    
    Args:
        tensor: Нормализованный тензор
        mean: Средние значения для денормализации
        std: Стандартные отклонения для денормализации
    
    Returns:
        Денормализованный тензор
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    return tensor * std + mean


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Применение CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: Входное изображение
        clip_limit: Лимит контраста
    
    Returns:
        Обработанное изображение
    """
    if len(image.shape) == 3:
        # Конвертируем в LAB и применяем CLAHE к L каналу
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return clahe.apply(image)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Вычисление основных метрик классификации.
    
    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
        y_prob: Предсказанные вероятности
    
    Returns:
        Словарь с метриками
    """
    # Основные метрики
    accuracy = accuracy_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob)
    
    # Confusion matrix для вычисления sensitivity и specificity
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)  # Recall
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    else:
        sensitivity = specificity = precision = f1_score = 0
    
    return {
        'accuracy': accuracy,
        'auc_roc': auc_roc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1_score
    }


def plot_training_history(train_losses: list, val_losses: list, 
                         train_accs: list, val_accs: list) -> plt.Figure:
    """
    Построение графиков обучения.
    
    Args:
        train_losses: Потери на обучении
        val_losses: Потери на валидации
        train_accs: Точность на обучении
        val_accs: Точность на валидации
    
    Returns:
        Figure с графиками
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # График потерь
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # График точности
    ax2.plot(train_accs, label='Train Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: list = ['Normal', 'Pneumonia']) -> plt.Figure:
    """
    Построение confusion matrix.
    
    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
        class_names: Названия классов
    
    Returns:
        Figure с confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    return fig


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> plt.Figure:
    """
    Построение ROC кривой.
    
    Args:
        y_true: Истинные метки
        y_prob: Предсказанные вероятности
    
    Returns:
        Figure с ROC кривой
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True)
    
    return fig


def save_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         epoch: int, loss: float, metrics: dict, filepath: str):
    """
    Сохранение checkpoint модели.
    
    Args:
        model: PyTorch модель
        optimizer: Оптимизатор
        epoch: Номер эпохи
        loss: Значение потерь
        metrics: Словарь с метриками
        filepath: Путь для сохранения
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }, filepath)
    print(f"Checkpoint сохранен: {filepath}")


def load_model_checkpoint(model: torch.nn.Module, filepath: str, 
                         optimizer: Optional[torch.optim.Optimizer] = None) -> dict:
    """
    Загрузка checkpoint модели.
    
    Args:
        model: PyTorch модель
        filepath: Путь к checkpoint
        optimizer: Оптимизатор (опционально)
    
    Returns:
        Словарь с информацией о checkpoint
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint не найден: {filepath}")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint загружен: {filepath}")
    return checkpoint


def ensure_dir(directory: str):
    """Создание директории если она не существует."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Создана директория: {directory}")


def print_metrics(metrics: dict, title: str = "Metrics"):
    """
    Красивый вывод метрик.
    
    Args:
        metrics: Словарь с метриками
        title: Заголовок
    """
    print(f"\n{title}")
    print("=" * len(title))
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric.capitalize():12}: {value:.4f}")
        else:
            print(f"{metric.capitalize():12}: {value}")
    print()


# Функция для установки seed для воспроизводимости
def set_random_seed(seed: int = 42):
    """Установка random seed для воспроизводимости результатов."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    # Тестирование функций
    print("Тестирование утилит...")
    
    # Тест предобработки изображения
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    processed = preprocess_image(test_image)
    print(f"Форма обработанного изображения: {processed.shape}")
    
    # Тест трансформаций
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()
    print("Трансформации созданы успешно")
    
    # Тест метрик
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    y_prob = np.array([0.1, 0.6, 0.8, 0.9, 0.2, 0.4])
    
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics, "Тестовые метрики")
    
    print("Все тесты пройдены успешно!")