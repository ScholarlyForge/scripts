import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from timm import create_model
from tqdm import tqdm
import kagglehub
import matplotlib.pyplot as plt

# ----- 데이터 다운로드 -----
DATA_PATH = kagglehub.dataset_download("arnaud58/flickrfaceshq-dataset-ffhq")
print('Dataset path:", DATA_PATH)

# ----- 하이퍼파라미터 후보 -----
param_grid = [
    {"BATCH_SIZE": 16, "LR": 3e-4, "EPOCHS": 8},
    {"BATCH_SIZE": 32, "LR": 1e-4, "EPOCHS": 10},
    {"BATCH_SIZE": 64, "LR": 5e-5, "EPOCHS": 12},
]

# ----- 설정 -----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
NUM_WORKERS = 4
MODEL_PATH = "best_vit_b32_ffhq.pth"

# ----- 데이터 전처리 -----
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

# ----- 데이터셋 로드 -----
train_dataset = datasets.ImageFolder(os.path.join(DATA_PATH, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_PATH, 'val'), transform=transform)

train_classes = train_dataset.classes
NUM_CLASSES = len(train_classes)
print(f' Detected classes: {train_classes}")

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

# ----- 최적 모델 저장용 -----
best_acc = 0.0
best_params = {}
results = []

# ----- 반복 실험 -----
for i, params in enumerate(param_grid):
    print(f"\n🧪 Trial {i + 1}/{len(param_grid)} - Params: {params}")

    # 데이터로더 설정
    train_loader = DataLoader(train_dataset, batch_size=params["BATCH_SIZE"], shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=params["BATCH_SIZE"], shuffle=False, num_workers=NUM_WORKERS)

    # 모델 초기화
    model = create_model('vit_base_patch32_224', pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["LR"])

    # 학습 반복
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

    # 기록 저장
    results.append({
        "BATCH_SIZE": params["BATCH_SIZE"],
        "LR": params["LR"],
        "EPOCHS": params["EPOCHS"],
        "ACC": acc
    })

    # 최고 성능 모델 저장
    if acc > best_acc:
        best_acc = acc
        best_params = params
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"New best model saved with acc {best_acc:.2f}% and params {best_params}")

print("\n전체 실험 완료!")
print(f"Best Accuracy: {best_acc:.2f}% with params: {best_params}")

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
