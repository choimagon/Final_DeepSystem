# 전체 코드 통합: train_crack_model 포함 투 스트림 학습
import os
import cv2
import numpy as np
import random
from PIL import Image
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from myCl import DualDeepLabV3Plus

OUTPUT_DIR = 'results_xception'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.0, eps=1e-7):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        batch_size = inputs.size(0)
        total_loss = 0

        for b in range(batch_size):
            input_b = inputs[b]
            target_b = targets[b]
            for c in range(input_b.size(0)):
                pred_c = input_b[c]
                target_c = (target_b == c).float()
                TP = (pred_c * target_c).sum()
                FP = ((1 - target_c) * pred_c).sum()
                FN = (target_c * (1 - pred_c)).sum()
                tversky = (TP + self.eps) / (TP + self.alpha * FN + self.beta * FP + self.eps)
                total_loss += (1 - tversky) ** self.gamma

        return total_loss / batch_size

class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir, files=None, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = files if files is not None else sorted([
            f for f in os.listdir(img_dir) if f.endswith(".jpg") or f.endswith(".png")
        ])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = np.array(Image.open(img_path).convert("RGB"))
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blur1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
        blur2 = cv2.GaussianBlur(gray, (0, 0), 2.0)
        dog = cv2.subtract(blur1, blur2)
        dog = np.clip(dog, 0, 255).astype(np.uint8)

        # 3채널로 확장
        if len(dog.shape) == 2:
            dog = np.stack([dog]*3, axis=2)  # [H, W] → [H, W, 3]

        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        green_mask = (mask_rgb[:, :, 0] < 50) & (mask_rgb[:, :, 1] > 200) & (mask_rgb[:, :, 2] < 50)
        binary_mask = (green_mask.astype(np.uint8) > 0).astype(np.uint8)

        if self.transform:
            rgb_trans = self.transform(image=image, mask=binary_mask)
            dog_trans = self.transform(image=dog, mask=binary_mask)
            rgb_tensor = rgb_trans["image"]
            dog_tensor = dog_trans["image"]
            mask_tensor = rgb_trans["mask"].long()
        else:
            rgb_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            dog_tensor = torch.from_numpy(dog.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(binary_mask).long()

        return (rgb_tensor, dog_tensor), mask_tensor

def calculate_iou(pred, target):
    pred, target = pred.bool(), target.bool()
    intersection = (pred & target).float().sum((1, 2))
    union = (pred | target).float().sum((1, 2))
    return ((intersection + 1e-6) / (union + 1e-6)).mean().item()

def setup_dataset_from_aihub(aihub_dir='aihub_dataset', validation_split=0.2):
    print(f"AI Hub 데이터셋 폴더 탐색 중: {aihub_dir}")
    
    train_normal_dir = os.path.join(aihub_dir, 'train', 'normal')
    train_masking_dir = os.path.join(aihub_dir, 'train', 'masking')
    val_normal_dir = os.path.join(aihub_dir, 'val', 'normal')
    val_masking_dir = os.path.join(aihub_dir, 'val', 'masking')
    
    if not os.path.exists(train_normal_dir) or not os.path.exists(train_masking_dir):
        print(f"오류: train 폴더를 찾을 수 없습니다. 경로: {train_normal_dir}")
        return None
    if not os.path.exists(val_normal_dir) or not os.path.exists(val_masking_dir):
        print(f"오류: val 폴더를 찾을 수 없습니다. 경로: {val_normal_dir}")
        return None
    
    train_files = sorted([f for f in os.listdir(train_normal_dir) if f.endswith('.jpg') or f.endswith('.png')])
    val_files = sorted([f for f in os.listdir(val_normal_dir) if f.endswith('.jpg') or f.endswith('.png')])
    
    print(f"발견된 train 이미지: {len(train_files)}개")
    print(f"발견된 test 이미지: {len(val_files)}개")
    
    random.shuffle(train_files)
    split_idx = int(len(train_files) * (1 - validation_split))
    train_split_files = train_files[:split_idx]
    val_split_files = train_files[split_idx:]
    
    print(f"학습 세트: {len(train_split_files)}개")
    print(f"검증 세트: {len(val_split_files)}개")
    print(f"테스트 세트: {len(val_files)}개")
    
    def compute_class_weights(files, mask_dir):
        print(f"클래스 가중치 계산 중...")
        total_counts = np.zeros(2)
        
        for fname in tqdm(files, desc="마스크 분석"):
            path = os.path.join(mask_dir, fname)
            if os.path.exists(path):
                mask_rgb = np.array(Image.open(path).convert("RGB"))
                green_mask = (mask_rgb[:, :, 0] < 50) & (mask_rgb[:, :, 1] > 200) & (mask_rgb[:, :, 2] < 50)
                binary_mask = (green_mask.astype(np.uint8) > 0).astype(np.uint8)
                
                unique, counts = np.unique(binary_mask, return_counts=True)
                for u, c in zip(unique, counts):
                    total_counts[u] += c
        
        total_counts = np.maximum(total_counts, 1)
        total = total_counts.sum()
        class_weights = total / (2.0 * total_counts)
        
        print("클래스 분포:")
        print(f"  - 배경: {int(total_counts[0])}픽셀 ({total_counts[0]/total*100:.2f}%)")
        print(f"  - 균열: {int(total_counts[1])}픽셀 ({total_counts[1]/total*100:.2f}%)")
        print(f"클래스 가중치: [{class_weights[0]:.4f}, {class_weights[1]:.4f}]")
        
        return class_weights
    
    class_weights = compute_class_weights(train_split_files[:min(1000, len(train_split_files))], train_masking_dir)
    
    return {
        'train_dataset': CrackDataset(train_normal_dir, train_masking_dir, train_split_files),
        'val_dataset': CrackDataset(train_normal_dir, train_masking_dir, val_split_files),
        'test_dataset': CrackDataset(val_normal_dir, val_masking_dir, val_files),
        'class_weights': class_weights.tolist()
    }


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_iou = 0, 0

    for (x_rgb, x_dog), masks in tqdm(loader, desc='Train'):
        x_rgb, x_dog, masks = x_rgb.to(device), x_dog.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(x_rgb, x_dog)
        # print(outputs.shape)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        iou = calculate_iou(preds, masks)
        total_iou += iou

    return total_loss / len(loader), total_iou / len(loader)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_iou = 0, 0

    with torch.no_grad():
        for (x_rgb, x_dog), masks in tqdm(loader, desc='Eval'):
            x_rgb, x_dog, masks = x_rgb.to(device), x_dog.to(device), masks.to(device)
            outputs = model(x_rgb, x_dog)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            iou = calculate_iou(preds, masks)
            total_iou += iou

    return total_loss / len(loader), total_iou / len(loader)

def visualize_prediction(x_rgb, x_dog, mask, pred):
    img = x_rgb.cpu().permute(1, 2, 0).numpy()
    img = img * 0.5 + 0.5
    mask = mask.cpu().numpy()
    pred = pred.cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img)
    axs[0].set_title('RGB Input')
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title('Ground Truth')
    axs[2].imshow(pred, cmap='gray')
    axs[2].set_title('Prediction')
    for ax in axs:
        ax.axis('off')
    plt.show()
    
def plot_training_results(train_losses, val_losses, train_ious, val_ious, output_dir):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_ious, label='Train IoU')
    plt.plot(epochs, val_ious, label='Val IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('IoU per Epoch')
    plt.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_results_combined.png')
    plt.savefig(save_path)
    plt.close()
    print(f"📊 학습 시각화 저장됨: {save_path}")

def train_crack_model(train_loader, val_loader, model, optimizer, criterion, device, num_epochs=30):
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_ious, val_ious = [], []

    for epoch in range(num_epochs):
        train_loss, train_iou = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_iou = eval_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)

        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"[Epoch {epoch+1}/{num_epochs}] Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print("✔ 모델 저장 완료")

    plot_training_results(train_losses, val_losses, train_ious, val_ious, OUTPUT_DIR)
    return model

def train_crack_model(train_loader, val_loader, model, optimizer, criterion, device, num_epochs=30):
    best_val_loss = float('inf')
    num_epochs = 1
    print(f"num_epochs = {num_epochs}")
    for epoch in range(num_epochs):
        train_loss, train_iou = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_iou = eval_epoch(model, val_loader, criterion, device)

        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"[Epoch {epoch+1}/{num_epochs}] Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print("✔ 모델 저장 완료")

    return model
def test_model(model, test_loader, device, criterion=None):
    model.eval()
    test_iou = 0
    test_loss = 0
    test_dice = 0
    test_accuracy = 0
    test_recall = 0
    
    if criterion is None:
        criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.0)

    def calculate_iou(pred, target):
        pred = pred.bool()
        target = target.bool()
        intersection = (pred & target).float().sum((1, 2))
        union = (pred | target).float().sum((1, 2))
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.mean().item()

    def calculate_recall(pred, target):
        pred = pred.bool()
        target = target.bool()
        true_positive = (pred & target).sum()
        false_negative = (~pred & target).sum()
        recall = (true_positive + 1e-6) / (true_positive + false_negative + 1e-6)
        return recall.item()
    
    with torch.no_grad():
        for (x_rgb, x_dog), masks in tqdm(test_loader, desc='테스트 중'):
            x_rgb = x_rgb.to(device)
            x_dog = x_dog.to(device)
            masks = masks.to(device)

            outputs = model(x_rgb, x_dog)
            loss = criterion(outputs, masks)
            test_loss += loss.item()

            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            iou_val = calculate_iou(preds, masks)
            test_iou += iou_val
            test_dice += (2 * iou_val) / (iou_val + 1)
            test_accuracy += (preds == masks).float().mean().item()
            test_recall += calculate_recall(preds, masks)

    test_loss /= len(test_loader)
    test_iou /= len(test_loader)
    test_dice /= len(test_loader)
    test_accuracy /= len(test_loader)
    test_recall /= len(test_loader)

    print(f'🧪 테스트 결과')
    print(f'  🔸 손실      : {test_loss:.4f}')
    print(f'  🔸 IoU       : {test_iou:.4f}')
    print(f'  🔸 Dice      : {test_dice:.4f}')
    print(f'  🔸 Accuracy  : {test_accuracy:.4f}')
    print(f'  🔸 Recall    : {test_recall:.4f}')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    datasets = setup_dataset_from_aihub('/home/magon/Downloads/TT/딥러닝 시스템')

    transform = A.Compose([
        A.Resize(299, 299),
        A.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ToTensorV2()
    ])

    datasets['train_dataset'].transform = transform
    datasets['val_dataset'].transform = transform

    train_loader = DataLoader(datasets['train_dataset'], batch_size=8, shuffle=True)
    val_loader = DataLoader(datasets['val_dataset'], batch_size=4, shuffle=False)

    model = DualDeepLabV3Plus(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = FocalTverskyLoss()

    train_crack_model(train_loader, val_loader, model, optimizer, criterion, device)
