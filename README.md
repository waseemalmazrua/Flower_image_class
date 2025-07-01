# 🌸 Flower Image Classifier

This project is a deep learning application that classifies flower images into one of 102 categories.  
It is built using PyTorch with a **fine-tuned VGG16 model**, and deployed via **Streamlit** for an interactive web interface.

---

##  What does it do?

- Classifies flower images into **102 flower species** from the Flowers102 dataset.
-  Displays the **top 3 predicted flower names** along with their probabilities.
-  Accepts any JPG/PNG flower image uploaded by the user via Streamlit.

---

## Tech stack

- **Python 3.11**
- **PyTorch & torchvision** - for deep learning & transfer learning
- **Streamlit** - for building the web app
- **PIL, NumPy** - image processing
- **Flowers102 dataset** - ~8,000 images, 102 classes

---

##  Model details

- Base model: `VGG16` pretrained on ImageNet.
- Fine-tuned on **Flowers102 dataset** (~6,000 training images).
- Classifies into **102 different flower species**.

![Screenshot 2025-07-01 213736](https://github.com/user-attachments/assets/1dd27f90-090d-4a39-80a7-5d5e09fc3126)







![Screenshot 2025-07-01 213809](https://github.com/user-attachments/assets/a0a2c91b-51b3-4167-90d0-842d415cf69d)


