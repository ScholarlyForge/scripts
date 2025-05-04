import os
import shutil
import random
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from timm import create_model
import kagglehub
import matplotlib.pyplot as plt

# ----- 기본 설정 -----
SEED = 42
random.seed(SEED)
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_DIR = os.getcwd()
DATASET_NAME = "ffhq-random-label"
CLASSES = ["male", "female"]
TRAIN_RATIO = 0.8
MODEL_PATH = "best_vit_b32_ffhq.pth"

# ----- 이미지 포함된 폴더 자동 탐색 함수 -----
def find_image_folder(root_path, valid_exts=(".jpg", ".png")):
    for root, _, files in os.walk(root_path):
        if any(file.lower().endswith(valid_exts) for file in files):
            return root
    return None

# ----- FFHQ 데이터 다운로드 -----
downloaded_path = kagglehub.dataset_download("arnaud58/flickrfaceshq-dataset-ffhq")
print("📦 Downloaded path:", downloaded_path)

source_img_path = find_image_folder(downloaded_path)
if source_img_path is None:
    raise RuntimeError(f"❌ No image folder found inside: {downloaded_path}")
print(f"🖼 Found image folder: {source_img_path}")

# ----- 자동 분류용 폴더 생성 -----
organized_path = os.path.join(PROJECT_DIR, DATASET_NAME)
for split in ["train", "val"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(organized_path, split, cls), exist_ok=True)

# ----- 이미지 복사 + 랜덤 라벨링 -----
image_files = [f for f in os.listdir(source_img_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
random.shuffle(image_files)
split_idx = int(len(image_files) * TRAIN_RATIO)

print("🛠 Organizing dataset into ImageFolder structure...")
for idx, img_file in enumerate(tqdm(image_files)):
    label = random.choice(CLASSES)  # 임의 라벨 부여
    split = "train" if idx < split_idx else "val"
    src = os.path.join(source_img_path, img_file)
    dst = os.path.join(organized_path, split, label, img_file)
    shutil.copyfile(src, dst)

# ----- transform & dataset -----
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

train_dataset = datasets.ImageFolder(os.path.join(organized_path, 'train'), transform=transform)
val_dataset   = datasets.ImageFolder(os.path.join(organized_path, 'val'), transform=transform)
class_names = train_dataset.classes
NUM_CLASSES = len(class_names)

print(f"✅ Detected classes: {class_names}")

# ----- 하이퍼파라미터 후보 -----
param_grid = [
    {"BATCH_SIZE": 16, "LR": 3e-4, "EPOCHS": 8},
    {"BATCH_SIZE": 32, "LR": 1e-4, "EPOCHS": 10},
    {"BATCH_SIZE": 64, "LR": 5e-5, "EPOCHS": 12},
]

# ----- 평가 함수 -----
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# ----- 반복 실험 -----
best_acc = 0.0
best_params = {}
results = []

for i, params in enumerate(param_grid):
    print(f"\n🧪 Trial {i + 1}/{len(param_grid)} - Params: {params}")

    train_loader = DataLoader(train_dataset, batch_size=params["BATCH_SIZE"], shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=params["BATCH_SIZE"], shuffle=False, num_workers=4)

    model = create_model('vit_base_patch32_224', pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["LR"])

    for epoch in range(params["EPOCHS"]):
        model.train()
        total_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        acc = evaluate(model, val_loader)
        print(f"📊 Epoch [{epoch + 1}/{params['EPOCHS']}], Loss: {total_loss:.4f}, Val Acc: {acc:.2f}%")

    results.append({
        "BATCH_SIZE": params["BATCH_SIZE"],
        "LR": params["LR"],
        "EPOCHS": params["EPOCHS"],
        "ACC": acc
    })

    if acc > best_acc:
        best_acc = acc
        best_params = params
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"🏅 Best model saved with acc {best_acc:.2f}% and params {best_params}")

# ----- 시각화 -----
labels = [f"BS={r['BATCH_SIZE']}, LR={r['LR']}, EP={r['EPOCHS']}" for r in results]
accuracies = [r["ACC"] for r in results]

plt.figure(figsize=(10, 6))
plt.bar(labels, accuracies, color='skyblue')
plt.ylabel("Validation Accuracy (%)")
plt.title("ViT-B32 FFHQ - Hyperparameter Tuning Results")
plt.ylim(0, 100)
plt.xticks(rotation=15)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
