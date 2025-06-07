import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from network.modeling import deeplabv3plus_xception

OUTPUT_DIR = 'results_xception'
os.makedirs(OUTPUT_DIR, exist_ok=True)

output_training_dir = os.path.join(OUTPUT_DIR, 'training_results.png')
output_test_dir = os.path.join(OUTPUT_DIR, 'test_predictions.png')

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.0, eps=1e-7):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha  # false negative penalty
        self.beta = beta    # false positive penalty
        self.gamma = gamma  # focusing parameter
        self.eps = eps

    def forward(self, inputs, targets):
        # softmax 적용
        inputs = F.softmax(inputs, dim=1)
        
        # 배치 차원 유지하면서 계산
        batch_size = inputs.size(0)
        total_loss = 0
        
        for b in range(batch_size):
            # 현재 배치의 입력과 타겟
            input_b = inputs[b]  # [C, H, W]
            target_b = targets[b]  # [H, W]
            
            # 각 클래스에 대해 Tversky 손실 계산
            for c in range(input_b.size(0)):
                # 현재 클래스의 예측과 타겟
                pred_c = input_b[c]  # [H, W]
                target_c = (target_b == c).float()  # [H, W]
                
                # TP, FP, FN 계산
                TP = (pred_c * target_c).sum()
                FP = ((1 - target_c) * pred_c).sum()
                FN = (target_c * (1 - pred_c)).sum()
                
                # Tversky 손실 계산
                tversky = (TP + self.eps) / (TP + self.alpha * FN + self.beta * FP + self.eps)
                focal_tversky = (1 - tversky) ** self.gamma
                
                total_loss += focal_tversky
        
        return total_loss / batch_size

# 데이터셋 설정 함수
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
    
    # 클래스 가중치 계산
    def compute_class_weights(files, mask_dir):
        print(f"클래스 가중치 계산 중...")
        total_counts = np.zeros(2)  # [배경, 균열]
        
        for fname in tqdm(files, desc="마스크 분석"):
            path = os.path.join(mask_dir, fname)
            if os.path.exists(path):
                mask_rgb = np.array(Image.open(path).convert("RGB"))
                green_mask = (mask_rgb[:, :, 0] < 50) & (mask_rgb[:, :, 1] > 200) & (mask_rgb[:, :, 2] < 50)
                binary_mask = (green_mask.astype(np.uint8) > 0).astype(np.uint8)
                
                unique, counts = np.unique(binary_mask, return_counts=True)
                for u, c in zip(unique, counts):
                    total_counts[u] += c
        
        total_counts = np.maximum(total_counts, 1)  # 0 나눗셈 방지
        total = total_counts.sum()
        class_weights = total / (2.0 * total_counts)
        
        print("클래스 분포:")
        print(f"  - 배경: {int(total_counts[0])}픽셀 ({total_counts[0]/total*100:.2f}%)")
        print(f"  - 균열: {int(total_counts[1])}픽셀 ({total_counts[1]/total*100:.2f}%)")
        print(f"클래스 가중치: [{class_weights[0]:.4f}, {class_weights[1]:.4f}]")
        
        return class_weights
    
    class_weights = compute_class_weights(train_split_files[:min(1000, len(train_split_files))], train_masking_dir)
    
    return {
        'train_files': train_split_files,
        'val_files': val_split_files,
        'test_files': val_files,
        'train_img_dir': train_normal_dir,
        'train_mask_dir': train_masking_dir,
        'val_img_dir': val_normal_dir,
        'val_mask_dir': val_masking_dir,
        'class_weights': class_weights.tolist() if isinstance(class_weights, np.ndarray) else class_weights
    }
# Dataset 클래스
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

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            raise FileNotFoundError(f"{img_path} 또는 {mask_path} 존재하지 않음")

        image = np.array(Image.open(img_path).convert("RGB"))
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))

        # 초록색 픽셀만 1로, 나머지는 0
        green_mask = (mask_rgb[:, :, 0] < 50) & (mask_rgb[:, :, 1] > 200) & (mask_rgb[:, :, 2] < 50)
        binary_mask = (green_mask.astype(np.uint8) > 0).astype(np.uint8)

        # transform 적용
        if self.transform:
            transformed = self.transform(image=image, mask=binary_mask)
            image = transformed["image"]
            mask_tensor = transformed["mask"].long()
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(binary_mask).long()

        return image, mask_tensor

# IoU 계산 함수
def calculate_iou(pred, target):
    pred = pred.bool()
    target = target.bool()
    intersection = (pred & target).float().sum((1, 2))
    union = (pred | target).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

# Recall 계산 함수
def calculate_recall(pred, target, eps=1e-6):
    pred = pred.bool()
    target = target.bool()
    true_positive = (pred & target).float().sum((1, 2))
    false_negative = (~pred & target).float().sum((1, 2))
    recall = (true_positive + eps) / (true_positive + false_negative + eps)
    return recall.mean().item()

# 학습 코드 수정
def train_crack_model():
    # 데이터셋 설정
    dataset_info = setup_dataset_from_aihub('/Users/khseon/Coding/4_1/DeepLabV3Plus-Pytorch/aihub_dataset')
    
    if not dataset_info:
        print("데이터셋 설정 실패")
        return
    
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")
    
    # 학습 데이터를 위한 augmentation이 포함된 transform
    train_transform = A.Compose([
        A.Resize(299,299),
        # 미세한 blur는 유효 (균열이 흐릿하게 보일 수도 있으므로)
        A.OneOf([
            A.Blur(blur_limit=3, p=0.3),
            A.MedianBlur(blur_limit=3, p=0.3),
            A.GaussianBlur(blur_limit=(3,5), p=0.3),
        ], p=0.5),
        
        # 대비 및 밝기 변환 (촬영 조건 변화 시나리오)
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),

        # 약간의 노이즈로 일반화 유도
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        
        # 좌우 반전 (균열 방향은 바뀔 수 있음)
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ])
    
    # Xception 모델을 위한 transforms 설정
    test_transform = A.Compose([
        A.Resize(299,299),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # [-1, 1] 범위로 정규화
        ToTensorV2()
    ])
    
    # 데이터셋 생성
    train_dataset = CrackDataset(
        dataset_info['train_img_dir'], 
        dataset_info['train_mask_dir'], 
        files=dataset_info['train_files'],
        transform=train_transform,
    )
    
    val_dataset = CrackDataset(
        dataset_info['train_img_dir'], 
        dataset_info['train_mask_dir'], 
        files=dataset_info['val_files'],
        transform=test_transform,
    )
    
    test_dataset = CrackDataset(
        dataset_info['val_img_dir'], 
        dataset_info['val_mask_dir'], 
        files=dataset_info['test_files'],
        transform=test_transform,
    )
    
    # 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    # 모델 초기화
    model = deeplabv3plus_xception(num_classes=2, output_stride=16)
    model = model.to(device)
    
    # 가중치 설정
    if dataset_info['class_weights']:
        class_weights = torch.tensor(dataset_info['class_weights'], dtype=torch.float32)
        print(f"클래스 가중치 사용: {class_weights}")
    else:
        class_weights = None
    
    # 손실 함수 및 옵티마이저
    # criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # 학습 파라미터
    num_epochs = 100
    best_val_loss = float('inf')
    early_stopping_patience = 10
    early_stopping_counter = 0
    
    # 결과 저장용 딕셔너리
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_iou': [],
        'val_iou': [],
        'train_dice': [],
        'val_dice': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'train_recall': [],
        'val_recall': []
    }
    
    # 학습 루프
    for epoch in range(num_epochs):
        # 학습 모드
        model.train()
        train_loss = 0
        train_iou = 0
        train_dice = 0
        train_accuracy = 0
        train_recall = 0
        
        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            images = images.to(device)
            masks = masks.to(device)
            
            # 그라디언트 초기화
            optimizer.zero_grad()
            
            # 순전파
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            # 손실 누적
            train_loss += loss.item()
            
            # IoU 계산 (softmax 적용 후)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
            iou_val = calculate_iou(preds, masks)
            train_iou += iou_val
            train_dice += (2*iou_val)/(iou_val+1)
            train_accuracy += (preds == masks).float().mean().item()
            train_recall += calculate_recall(preds, masks)
        # 에포크당 평균 손실 및 IoU
        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        train_dice /= len(train_loader)
        train_accuracy /= len(train_loader)
        train_recall /= len(train_loader)
        
        # 검증 모드
        model.eval()
        val_loss = 0
        val_iou = 0
        val_dice = 0
        val_accuracy = 0
        val_recall = 0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                images = images.to(device)
                masks = masks.to(device)
                
                # 순전파
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # 손실 누적
                val_loss += loss.item()
                
                # IoU 계산
                preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
                iou_val = calculate_iou(preds, masks)
                val_iou += iou_val
                val_dice += (2*iou_val)/(iou_val+1)
                val_accuracy += (preds == masks).float().mean().item()
                val_recall += calculate_recall(preds, masks)
        
        # 에포크당 평균 검증 손실 및 IoU
        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_dice /= len(val_loader)
        val_accuracy /= len(val_loader)
        val_recall /= len(val_loader)
        
        # 학습률 스케줄러
        scheduler.step(val_loss)
        
        # 결과 저장
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        history['train_recall'].append(train_recall)
        history['val_recall'].append(val_recall)
        
        # 에포크 결과 출력
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, Train Dice: {train_dice:.4f}, Train Accuracy: {train_accuracy:.4f}, Train Recall: {train_recall:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, Val Dice: {val_dice:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Recall: {val_recall:.4f}')
        
        # 최적 모델 저장
        if val_loss < best_val_loss:
            early_stopping_counter = 0
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'iou': val_iou,
                'dice': val_dice,
                'accuracy': val_accuracy,
                'recall': val_recall
            }, os.path.join(OUTPUT_DIR,f'xception_cbam_before_ASPP_{best_val_loss:.4f}_{val_iou:.4f}_{val_dice:.4f}_{val_accuracy:.4f}_{val_recall:.4f}.pth'))
            print(f'모델이 저장되었습니다! (Epoch {epoch+1})')
        
        # Early Stopping 체크
        if val_loss > best_val_loss:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early Stopping at Epoch {epoch+1}")
                break
    
    # 학습 결과 시각화
    plot_training_results(history)
    
    # 테스트 세트 평가
    print("\n테스트 세트 평가:")
    test_model(model, test_loader, device, criterion)
    
    return model, history

def plot_training_results(history):
    plt.figure(figsize=(15, 5))
    
    # 손실 그래프
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # IoU 그래프
    plt.subplot(1, 3, 2)
    plt.plot(history['train_iou'], label='Train IoU')
    plt.plot(history['val_iou'], label='Validation IoU')
    plt.title('IoU per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history['train_dice'], label='Train Dice')
    plt.plot(history['val_dice'], label='Validation Dice')
    plt.title('Dice per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_training_dir)

def visualize_predictions(model, test_loader, device, num_samples=5):
    model.eval()
    
    # 랜덤 배치 선택
    dataiter = iter(test_loader)
    images, masks = next(dataiter)
    
    # 샘플 수 제한
    num_samples = min(num_samples, images.size(0))
    
    images = images[:num_samples].to(device)
    masks = masks[:num_samples].to(device)
    
    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
    
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    
    for i in range(num_samples):
        # 원본 이미지 - 정규화 복원
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = img * 0.5 + 0.5  # [-1,1] -> [0,1]
        
        # 실제 마스크
        mask = masks[i].cpu().numpy()
        
        # 예측 마스크
        pred = preds[i].cpu().numpy()
        
        # 시각화
        axs[i, 0].imshow(img)
        axs[i, 0].set_title('Input image')
        axs[i, 0].axis('off')
        
        axs[i, 1].imshow(mask, cmap='gray')
        axs[i, 1].set_title('True Mask')
        axs[i, 1].axis('off')
        
        axs[i, 2].imshow(pred, cmap='gray')
        axs[i, 2].set_title('Predicted Mask')
        axs[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_test_dir)

def test_model(model, test_loader, device, criterion=None):
    model.eval()
    test_iou = 0
    test_loss = 0
    test_dice = 0
    test_accuracy = 0
    test_recall = 0
    
    if not criterion:
        criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.0)
    
    def calculate_iou(pred, target):
        pred = pred.bool()
        target = target.bool()
        intersection = (pred & target).float().sum((1, 2))
        union = (pred | target).float().sum((1, 2))
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.mean().item()
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc='테스트 중'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item()
            
            # Binary segmentation을 위한 sigmoid 적용
            probs = torch.sigmoid(outputs[:, 1])
            preds = (probs > 0.5).float()
            iou_val = calculate_iou(preds, masks)
            test_iou += iou_val
            test_dice += (2*iou_val)/(iou_val+1)
            test_accuracy += (preds == masks).float().mean().item()
            test_recall += calculate_recall(preds, masks)
            
    test_loss /= len(test_loader)
    test_iou /= len(test_loader)
    test_dice /= len(test_loader)
    test_accuracy /= len(test_loader)
    test_recall /= len(test_loader)
    
    print(f'테스트 손실: {test_loss:.4f}, 테스트 IoU: {test_iou:.4f}, 테스트 Dice: {test_dice:.4f}, 테스트 Accuracy: {test_accuracy:.4f}, 테스트 Recall: {test_recall:.4f}')
    
    # 테스트 결과 시각화
    visualize_predictions(model, test_loader, device)
    
    return test_loss, test_iou, test_dice, test_accuracy, test_recall

if __name__ == "__main__":
    train_crack_model()