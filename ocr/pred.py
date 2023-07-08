#! /usr/bin/env python3

import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from ocr import Classifier


# 加载模型
model = Classifier()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# 定义transform
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# 读取图像
img_path = "9.jpg"  # 这里替换为你的图像路径
image = Image.open(img_path)

# 将图像转为模型输入
input_image = transform(image)
input_image = input_image.unsqueeze(0)  # 添加batch维度

# 预测
with torch.no_grad():
    logps = model(input_image)
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))
    print("Predicted Class : ", pred_label)
