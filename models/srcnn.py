import torch
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        # Первый сверточный слой (Feature Extraction)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(64)  # Добавили BatchNorm

        # Второй сверточный слой (Non-linear Mapping)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(32)  # Добавили BatchNorm

        # Третий сверточный слой (Reconstruction)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))  # Применяем BatchNorm
        x = self.relu2(self.bn2(self.conv2(x)))  # Применяем BatchNorm
        x = torch.sigmoid(self.conv3(x))  # Ограничиваем выход в диапазоне [0,1]
        return x

if __name__ == "__main__":
    model = SRCNN()
    print(model)  # Вывод структуры модели