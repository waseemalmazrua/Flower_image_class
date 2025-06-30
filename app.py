import os
import gdown
import streamlit as st
import torch
from torchvision import models
from PIL import Image
import numpy as np
import json

# -------------------------
# Download checkpoint if missing
# -------------------------
def download_checkpoint():
    file_id = "1H-i3wU4VoloC59C_g83z2_HDz5PhC_uJ"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "vgg16_checkpoint.pth"
    if not os.path.exists(output):
        print(f"Downloading {output} from Google Drive...")
        gdown.download(url, output, quiet=False)
        print("Download completed.")
    else:
        print(f"{output} already exists locally.")

download_checkpoint()

# -------------------------
# Load model checkpoint
# -------------------------
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
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

# -------------------------
# Process image
# -------------------------
def process_image(image):
    if image.size[0] > image.size[1]:
        image.thumbnail((10000, 256))
    else:
        image.thumbnail((256, 10000))
    left = (image.width - 224) / 2
    top = (image.height - 224) / 2
    right = left + 224
    bottom = top + 224
    image = image.crop((left, top, right, bottom))
    np_image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    return torch.from_numpy(np_image).float()

# -------------------------
# Predict
# -------------------------
def predict(image, model, topk=3, device='cpu'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        img = process_image(image)
        img = img.unsqueeze(0).to(device)
        output = model(img)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
    return top_p.cpu().numpy()[0], top_class.cpu().numpy()[0]

# -------------------------
# Load model & category names
# -------------------------
model = load_checkpoint('vgg16_checkpoint.pth')
model.eval()

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# -------------------------
# Streamlit UI
# -------------------------
st.title("🌸 Flower Classifier App")
st.write("Upload a flower image and the model will predict the most likely flower species.")

uploaded_file = st.file_uploader("Upload a flower image...", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    probs, classes = predict(image, model, topk=3, device='cpu')
    names = [cat_to_name.get(str(cls + 1), 'Unknown') for cls in classes]

    st.write("### Predictions:")
    for i in range(len(probs)):
        st.write(f"{i+1}: **{names[i]}** ({probs[i]*100:.2f}%)")

st.write("""
**Built by Waseem Almazrua**  
[LinkedIn Profile](https://www.linkedin.com/in/waseemalmazrua/)

This app uses a fine-tuned VGG16 model trained on ~8,000 flower images (Flowers102 dataset)
to classify images into 102 flower categories.
""")
