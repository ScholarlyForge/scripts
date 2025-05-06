import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from timm import create_model
import torch.optim as optim
from tqdm import tqdm
import torchattacks
import multiprocessing

# ----- 공격 함수 정의 -----
criterion = nn.CrossEntropyLoss()

def fgsm_attack(img, label, model, eps):
    # FGSM requires grad, ensure enabled
    img_adv = img.clone().detach().requires_grad_(True)
    output = model(img_adv)
    loss = criterion(output, label)
    loss.backward()
    adv = img_adv + eps * img_adv.grad.sign()
    return adv.clamp(-1, 1).detach()


def pgd_attack(img, label, model, eps, alpha, steps):
    # PGD requires grad
    adv = img.clone().detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        output = model(adv)
        loss = criterion(output, label)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.min(torch.max(adv, img - eps), img + eps)
    return adv.clamp(-1, 1).detach()

cw_attack_fn = lambda m: torchattacks.CW(m, c=1e-2, steps=100)

# ----- 메인 함수 정의 -----
def main():
    # ----- 설정 -----
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE = 224
    NUM_CLASSES = 10
    BATCH_SIZE = 128
    EPOCHS = 15
    LEARNING_RATE = 3e-4
    SEED = 42

    # FGSM/PGD 파라미터
    EPS_FGSM = 0.01
    EPS_PGD = 0.01
    ALPHA_PGD = 0.003
    PGD_STEPS = 7

    # 저장 디렉터리 생성
    os.makedirs("./results", exist_ok=True)
    torch.manual_seed(SEED)

    # ----- 데이터 로드 및 전처리 -----
    transform_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])
    transform_test = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])
    train_dataset = datasets.CIFAR10(root="./data", train=True, transform=transform_train, download=True)
    val_dataset = datasets.CIFAR10(root="./data", train=False, transform=transform_test, download=True)
    # Windows: num_workers=0로 설정
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ----- 학습 설정: 8가지 케이스 -----
    cases = {
        "Clean": [],
        "FGSM": ["FGSM"],
        "PGD": ["PGD"],
        "CW": ["CW"],
        "FGSM+PGD": ["FGSM", "PGD"],
        "FGSM+CW": ["FGSM", "CW"],
        "PGD+CW": ["PGD", "CW"],
        "FGSM+PGD+CW": ["FGSM", "PGD", "CW"]
    }

    results = {}

    for case_name, attacks in cases.items():
        print(f"\n=== Training with {case_name} {'(clean)' if not attacks else 'adversarial'} training ===")
        model = create_model("vit_base_patch32_224", pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        cw_attack = cw_attack_fn(model) if "CW" in attacks else None

        # Training loop
        for epoch in range(1, EPOCHS + 1):
            model.train()
            pbar = tqdm(train_loader, desc=f"{case_name} Ep{epoch}")
            for imgs, labels in pbar:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                # Generate training images
                if not attacks:
                    train_imgs = imgs
                else:
                    adv_list = []
                    for atk in attacks:
                        if atk == "FGSM":
                            with torch.enable_grad():
                                adv_list.append(fgsm_attack(imgs, labels, model, EPS_FGSM))
                        elif atk == "PGD":
                            with torch.enable_grad():
                                adv_list.append(pgd_attack(imgs, labels, model, EPS_PGD, ALPHA_PGD, PGD_STEPS))
                        elif atk == "CW":
                            adv_list.append(cw_attack(imgs, labels))
                    train_imgs = torch.stack(adv_list).mean(0)

                optimizer.zero_grad()
                outputs = model(train_imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                pbar.set_postfix(loss=float(loss))

        # Save model
        torch.save(model.state_dict(), f"./results/{case_name}.pth")

        # Evaluation
        model.eval()
        total = clean_corr = 0
        adv_corr = {"FGSM": 0, "PGD": 0, "CW": 0}
        for imgs, labels in tqdm(val_loader, desc=f"Eval {case_name}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            total += labels.size(0)
            # Clean accuracy
            with torch.no_grad():
                clean_corr += (model(imgs).argmax(1) == labels).sum().item()
            # Adversarial accuracies
            # FGSM
            adv_fgsm = fgsm_attack(imgs, labels, model, EPS_FGSM)
            with torch.no_grad():
                adv_corr["FGSM"] += (model(adv_fgsm).argmax(1) == labels).sum().item()
            # PGD
            adv_pgd = pgd_attack(imgs, labels, model, EPS_PGD, ALPHA_PGD, PGD_STEPS)
            with torch.no_grad():
                adv_corr["PGD"] += (model(adv_pgd).argmax(1) == labels).sum().item()
            # CW
            if cw_attack:
                adv_cw = cw_attack(imgs, labels)
            else:
                adv_cw = imgs
            with torch.no_grad():
                adv_corr["CW"] += (model(adv_cw).argmax(1) == labels).sum().item()

        clean_acc = 100 * clean_corr / total
        atk_accs = {atk: 100 * adv_corr[atk] / total for atk in adv_corr}
        results[case_name] = {'clean': clean_acc, **atk_accs}
        print(f"{case_name}: Clean={clean_acc:.2f}%, FGSM={atk_accs['FGSM']:.2f}%, PGD={atk_accs['PGD']:.2f}%, CW={atk_accs['CW']:.2f}%")

    # Final summary
    print("\n=== Final Summary ===")
    for case, accs in results.items():
        print(f"{case}: Clean={accs['clean']:.2f}%, FGSM={accs['FGSM']:.2f}%, PGD={accs['PGD']:.2f}%, CW={accs['CW']:.2f}%")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
