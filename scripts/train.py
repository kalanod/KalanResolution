import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
from pathlib import Path
import numpy as np

from models.srcnn import SRCNN

def load_image(path, size=(128, 128)):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)  # Изменили интерполяцию
    img = img.astype(np.float32) / 127.5 - 1.0  # Преобразуем в диапазон [-1, 1]
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    return torch.tensor(img, dtype=torch.float32)

# Датасет
class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        lr_files = sorted(Path(lr_dir).glob("*.jpg")) + sorted(Path(lr_dir).glob("*.png"))
        hr_files = sorted(Path(hr_dir).glob("*.jpg")) + sorted(Path(hr_dir).glob("*.png"))
        self.pairs = [(str(lr), str(hr)) for lr, hr in zip(lr_files, hr_files)]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        lr_path, hr_path = self.pairs[index]
        return load_image(lr_path), load_image(hr_path)

# Метрика PSNR
def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(1.0 / math.sqrt(mse))

# Улучшенная SRCNN
class SRCNN_Improved(SRCNN):
    def __init__(self):
        super(SRCNN_Improved, self).__init__()
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.prelu(self.bn2(self.conv2(x)))
        x = torch.sigmoid(self.conv3(x))
        return x

def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        total_psnr = 0
        for lr, hr in train_loader:
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)

            optimizer.zero_grad()
            output = model(lr)
            loss = criterion(output, hr)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            total_psnr += psnr(output, hr)

        avg_loss = total_loss / len(train_loader)
        avg_psnr = total_psnr / len(train_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.6f}, PSNR: {avg_psnr:.2f} dB")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/srcnn_epoch{epoch + 1}.pth")

    print("Обучение завершено!")
    torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/srcnn.pth")

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True  # Ускорение вычислений на GPU

    # Пути
    DATA_DIR = "../data/train"
    CHECKPOINT_DIR = "../checkpoints"
    EPOCHS = 50
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    GRAD_CLIP = 0.1  # Ограничение градиента
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Создаем папку для чекпоинтов
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Загружаем данные
    train_dataset = SRDataset(f"{DATA_DIR}/lr", f"{DATA_DIR}/hr")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Модель, функция потерь, оптимизатор
    model = SRCNN_Improved().to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    train()
