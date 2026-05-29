
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# כיבוי אזהרות ה-Transformers שמתנגשות עם PyTorch 2.0.1
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

sys.path.append('..')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model (bypassing Transformers issues)...")

try:
    # נטען את המודל ישירות דרך open_clip בדיוק כמו ש-clip_benchmark עושה מאחורי הקלעים
    import open_clip
    model, _, transform = open_clip.create_model_and_transforms("musk_large_patch16_384", pretrained="models.txt", device=device)
except Exception as e:
    print(f"Open_clip load failed, falling back to local files... Error: {e}")
    # אם זה לא עובד, נטען ישירות מהקוד המקומי בתיקייה
    import importlib
    musk_module = importlib.import_module("musk.models.musk")
    model = getattr(musk_module, 'musk_large_patch16_384')(pretrained=True).to(device)
    transform = None

if transform is None:
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

dataset_path = "C:/Users/97250/Desktop/שנה ג/סמסטר ב/אונקולוגיה/הדגמה 1/musk/downstreams_demo/downstreams_demo/unitopatho"
if os.path.exists(os.path.join(dataset_path, "train")):
    dataset_path = os.path.join(dataset_path, "train")

dataset = ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

model.eval()
embeddings_list = []
labels_list = []

print("Extracting features (this might take a minute)...")
with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)
        if hasattr(model, 'encode_image'):
            features = model.encode_image(images)
        elif hasattr(model, 'forward_features'):
            features = model.forward_features(images)
            if len(features.shape) == 3:
                features = features.mean(dim=1)
        else:
            features = model(images)
            if len(features.shape) == 3:
                features = features.mean(dim=1)

        features = features / features.norm(dim=-1, keepdim=True)
        embeddings_list.append(features.cpu().numpy())
        labels_list.append(labels.numpy())

embeddings = np.concatenate(embeddings_list, axis=0)
labels = np.concatenate(labels_list, axis=0)
class_names = dataset.classes

print("Running PCA...")
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#1f77b4', '#8c564b', '#e377c2']

for i, class_label in enumerate(np.unique(labels)):
    mask = (labels == class_label)
    name = class_names[class_label] if class_label < len(class_names) else str(class_label)
    ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=name, color=colors[i % len(colors)], alpha=0.6, edgecolors='w', s=45)

var = pca.explained_variance_ratio_.sum() * 100
ax.set_title(f'MUSK Latent Space PCA (Unitopatho)\nExplained Variance: {var:.1f}%', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('PC 1', fontsize=11)
ax.set_ylabel('PC 2', fontsize=11)
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(loc='best', fontsize=10)

plt.tight_layout()
plt.savefig('latent_space_innovation.png', dpi=300)
print("Saved 'latent_space_innovation.png' successfully! Check your folder.")
