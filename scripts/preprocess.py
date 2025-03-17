import os
import cv2
import glob
import random
import shutil
from pathlib import Path

# Параметры
HR_DIR = "../data/hr"  # Исходные изображения высокого разрешения
LR_DIR = "../data/lr"  # Уменьшенные версии
SCALE_FACTOR = 2  # Во сколько раз уменьшаем разрешение
TRAIN_RATIO = 0.8  # Доля обучающего множества
TARGET_SIZE = 256  # Размер для обрезки и изменения разрешения

def crop_center_square(img):
    """Обрезает изображение до квадратного центра."""
    h, w, _ = img.shape
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    return img[top:top + min_dim, left:left + min_dim]

def create_lr_image(hr_path, lr_path, scale_factor):
    """Генерирует изображение низкого разрешения и сохраняет его."""
    img = cv2.imread(hr_path)
    img = crop_center_square(img)
    img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_CUBIC)
    lr_img = cv2.resize(img, (TARGET_SIZE // scale_factor, TARGET_SIZE // scale_factor), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(lr_path, lr_img)
    return img

def prepare_data():
    """Разбивает данные на train/val и создает LR изображения."""
    hr_images = glob.glob(f"{HR_DIR}/*.jpg") + glob.glob(f"{HR_DIR}/*.png")
    random.shuffle(hr_images)

    train_size = int(len(hr_images) * TRAIN_RATIO)
    train_hr, val_hr = hr_images[:train_size], hr_images[train_size:]

    for dataset, images in [("train", train_hr), ("val", val_hr)]:
        Path(f"../data/{dataset}/hr").mkdir(parents=True, exist_ok=True)
        Path(f"../data/{dataset}/lr").mkdir(parents=True, exist_ok=True)

        for hr_path in images:
            filename = os.path.basename(hr_path)
            lr_path = f"../data/{dataset}/lr/{filename}"
            hr_dest_path = f"../data/{dataset}/hr/{filename}"

            hr_img = create_lr_image(hr_path, lr_path, SCALE_FACTOR)
            cv2.imwrite(hr_dest_path, hr_img)  # Сохраняем HR-изображение с фиксированным размером

    print("Данные подготовлены!")

if __name__ == "__main__":
    prepare_data()
