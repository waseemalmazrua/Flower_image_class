import argparse
import torch
from torchvision import models
from PIL import Image
import json
import numpy as np

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    model = models.vgg16(pretrained=True)  # ثابت لأنك دربت على VGG16
    for param in model.parameters():
        param.requires_grad = False
    # هنا نبني نفس الطبقات التي دربت عليها في Part 1
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(25088, 4096),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(4096, 102),
        torch.nn.LogSoftmax(dim=1)
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image_path):
    image = Image.open(image_path)
    if image.size[0] > image.size[1]:
        image.thumbnail((10000, 256))
    else:
        image.thumbnail((256, 10000))
    left = (image.width - 224)/2
    top = (image.height - 224)/2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))
    np_image = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    return torch.from_numpy(np_image).float()

def predict(image_path, model, topk, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        img = process_image(image_path)
        img = img.unsqueeze(0).to(device)
        output = model(img)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
    return top_p.cpu().numpy()[0], top_class.cpu().numpy()[0]

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument('image_path')
parser.add_argument('checkpoint')
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--category_names', type=str, default=None)
parser.add_argument('--gpu', action='store_true')
args = parser.parse_args()

device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

model = load_checkpoint(args.checkpoint)
probs, classes = predict(args.image_path, model, args.top_k, device)

if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    names = [cat_to_name.get(str(cls+1), 'Unknown') for cls in classes]
else:
    names = classes

for i in range(args.top_k):
    print(f"{i+1}: Class: {names[i]}, Probability: {probs[i]:.3f}")
