import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from timm import create_model
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torchattacks

# ----- 설정 -----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
NUM_CLASSES = 10
MODEL_PATH = "best_vit_cifar10.pth"
SEED = 42
torch.manual_seed(SEED)

# 저장 디렉터리 생성
SAVE_DIR_SINGLE = "./attack_examples/single"
SAVE_DIR_COMBO = "./attack_examples/combo"
os.makedirs(SAVE_DIR_SINGLE, exist_ok=True)
os.makedirs(SAVE_DIR_COMBO, exist_ok=True)

# 클래스 이름
class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,) * 3, (0.5,) * 3)
])

val_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

# 모델 로드
model = create_model("vit_base_patch32_224", pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("✅ Pretrained model loaded!")
else:
    print("❌ Trained model not found.")
    exit()

# 마스크 생성 함수
def get_mask(shape, region):
    _, _, h, w = shape
    mask = torch.zeros((1, 3, h, w))
    half_h, half_w = h // 2, w // 2

    if region == "top_left":
        mask[:, :, :half_h, :half_w] = 1
    elif region == "top_right":
        mask[:, :, :half_h, half_w:] = 1
    elif region == "bottom_left":
        mask[:, :, half_h:, :half_w] = 1
    elif region == "bottom_right":
        mask[:, :, half_h:, half_w:] = 1
    elif region == "center":
        ch, cw = h // 4, w // 4
        mask[:, :, ch:ch*3, cw:cw*3] = 1
    elif region == "border":
        border = 16
        mask[:, :, :border, :] = 1
        mask[:, :, -border:, :] = 1
        mask[:, :, :, :border] = 1
        mask[:, :, :, -border:] = 1
    else:  # full
        mask[:] = 1
    return mask

# 공격용 손실 함수
criterion = nn.CrossEntropyLoss()

# FGSM 영역 공격
def fgsm_region(img, label, model, eps, mask):
    img_adv = img.clone().detach().requires_grad_(True)
    output = model(img_adv)
    loss = criterion(output, label)
    loss.backward()
    grad_sign = img_adv.grad.sign()
    adv = img_adv + eps * grad_sign * mask
    return adv.clamp(-1, 1).detach()

# PGD 영역 공격
def pgd_region(img, label, model, eps, alpha, steps, mask):
    adv = img.clone().detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        output = model(adv)
        loss = criterion(output, label)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign() * mask
        # ε-ball 내, 정규화 범위 내로 클램프
        adv = torch.min(torch.max(adv, img - eps), img + eps)
        adv = adv.clamp(-1, 1)
    return adv.detach()

# 시각화 및 저장 함수
def visualize_and_save_batch(orig_imgs, adv_imgs, orig_preds, adv_preds, label_ids, title):
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(title, fontsize=16)

    for i in range(10):
        row, col = divmod(i, 5)
        ax = axes[row][col]

        orig = orig_imgs[i].squeeze().cpu() * 0.5 + 0.5
        adv = adv_imgs[i].squeeze().cpu() * 0.5 + 0.5
        merged = torch.cat([orig, adv], dim=2)
        merged_np = merged.detach().permute(1, 2, 0).numpy()

        ax.imshow(merged_np)
        ax.set_title(f"{class_names[label_ids[i]]}\n{class_names[orig_preds[i]]}→{class_names[adv_preds[i]]}", fontsize=9)
        ax.axis("off")

        img_title = title.replace(" ", "_").replace("|", "_").replace(".", "")
        path_single = os.path.join(SAVE_DIR_SINGLE, f"{img_title}_{i}.png")
        plt.imsave(path_single, merged_np)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    combo_path = os.path.join(SAVE_DIR_COMBO, f"{title.replace(' ', '_').replace('|', '_').replace('.', '')}.png")
    plt.savefig(combo_path)
    plt.show()

# 공격 설정
epsilons = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
regions = ["full", "top_left", "top_right", "bottom_left", "bottom_right", "center", "border"]

for eps in epsilons:
    print(f"\n==== ε = {eps:.4f} ====")
    # CW는 full에서만
    cw_attack = torchattacks.CW(model, c=1e-2, steps=100)

    for region in regions:
        for atk_name in ["FGSM", "PGD", "CW"]:
            if atk_name == "CW" and region != "full":
                continue

            print(f"\n🚨 {atk_name} - {region}")
            correct = total = shown = 0
            orig_imgs, adv_imgs, orig_preds, adv_preds, label_ids = [], [], [], [], []

            for img, label in tqdm(val_dataset, desc=f"{atk_name}-{region}"):
                img = img.unsqueeze(0).to(DEVICE)
                label_tensor = torch.tensor([label]).to(DEVICE)
                mask = get_mask(img.shape, region).to(DEVICE)

                if atk_name == "FGSM":
                    adv = fgsm_region(img, label_tensor, model, eps, mask)
                elif atk_name == "PGD":
                    adv = pgd_region(img, label_tensor, model, eps, eps/4, 10, mask)
                else:  # CW
                    adv = cw_attack(img, label_tensor)

                with torch.no_grad():
                    orig_pred = model(img).argmax(1).item()
                    adv_pred = model(adv).argmax(1).item()
                    correct += (adv_pred == label)
                    total += 1

                if shown < 10:
                    orig_imgs.append(img)
                    adv_imgs.append(adv)
                    orig_preds.append(orig_pred)
                    adv_preds.append(adv_pred)
                    label_ids.append(label)
                    shown += 1

            acc = 100 * correct / total
            print(f"✨ Accuracy after attack: {acc:.2f}%")
            title = f"[{atk_name}] {region} | ε={eps:.4f} | Acc={acc:.2f}%"
            visualize_and_save_batch(orig_imgs, adv_imgs, orig_preds, adv_preds, label_ids, title)
