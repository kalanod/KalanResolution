import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

from scripts.train import SRCNN_Improved
import os
print("Файл весов существует:", os.path.exists("checkpoints/srcnn.pth"))

# Загрузка обученной модели
model = SRCNN_Improved()  # Используем ту же модель, что и при обучении
model.load_state_dict(torch.load("checkpoints/srcnn.pth", map_location=torch.device('cpu')))
model.eval()


def process_image(file):
    image = Image.open(file).convert('RGB')

    # Обрезка до квадрата
    width, height = image.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim
    image = image.crop((left, top, right, bottom))

    # Изменение размера до 256x256
    image = image.resize((256, 256), Image.LANCZOS)

    img_tensor = ToTensor()(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)

    output_image = ToPILImage()(output.squeeze())
    return output_image
