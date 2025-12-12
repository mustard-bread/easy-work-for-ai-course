import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 兼容新旧版本 PyTorch 的 AMP
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import sys

# ================= 核心配置区域 =================
CONFIG = {
    # 数据集路径
    "train_dir": r"C:\Users\Administrator\PycharmProjects\geo_detect\geometry_dataset_final\train",
    "val_dir": r"C:\Users\Administrator\PycharmProjects\geo_detect\geometry_dataset_final\val",

    # 预训练权重路径 (新练时用)
    "pretrained_path": r"C:\Users\Administrator\PycharmProjects\geo_detect\pytorch_model.bin",

    # === 【断点续练配置】 ===
    # 填写你想要加载的存档文件名 (必须在当前目录下)
    "resume_checkpoint": r"C:\Users\Administrator\PycharmProjects\geo_detect\geometry_dataset_final\last_model.pth",

    # 填写你要从第几轮开始 (比如你跑完了Epoch 4，这里就填 4，代表从第5轮开始)
    "start_epoch": 10,
    # ========================

    "img_size": 384,
    "batch_size": 2,
    "accumulate_grad": 16,
    "learning_rate": 5e-5,
    "epochs": 15,  # 总共要练多少轮
    "num_workers": 0,  # Windows 建议设为 0
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


# ===============================================

class GeometryDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.img_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")

        if not os.path.exists(self.img_dir):
            print(f"[Error] 找不到路径: {self.img_dir}")
            sys.exit(1)

        self.ids = [os.path.splitext(f)[0] for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        file_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{file_id}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{file_id}.png")

        # 读取图片 (BGR -> RGB)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 读取 Mask
        mask = cv2.imread(mask_path)

        # 重组通道: BGR -> RGB顺序 (Lines, Circles, Points)
        masks = np.stack([
            mask[:, :, 2],  # Lines
            mask[:, :, 1],  # Circles
            mask[:, :, 0]  # Points
        ], axis=-1).astype(np.float32) / 255.0

        if self.transform:
            augmented = self.transform(image=image, mask=masks)
            image = augmented['image']
            masks = augmented['mask']

            # 【维度修复】确保 Mask 是 [3, 384, 384]
        if isinstance(masks, torch.Tensor):
            if masks.shape[-1] == 3 and masks.shape[0] != 3:
                masks = masks.permute(2, 0, 1)
        elif isinstance(masks, np.ndarray):
            if masks.shape[-1] == 3:
                masks = np.transpose(masks, (2, 0, 1))
            masks = torch.from_numpy(masks)

        return image, masks


def get_transforms(phase="train"):
    trans = [
        A.Resize(CONFIG["img_size"], CONFIG["img_size"]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    if phase == "train":
        return A.Compose([
                             A.HorizontalFlip(p=0.5),
                             A.VerticalFlip(p=0.5),
                             A.Rotate(limit=15, p=0.5),
                             A.RandomBrightnessContrast(p=0.2)
                         ] + trans)
    return A.Compose(trans)


def train_fn(loader, model, optimizer, scaler, scheduler, loss_fn):
    model.train()
    loop = tqdm(loader, desc="Training")
    epoch_loss = 0

    for i, (data, targets) in enumerate(loop):
        data = data.to(CONFIG["device"])
        targets = targets.to(CONFIG["device"])

        # 混合精度
        if torch.cuda.is_available():
            with autocast('cuda'):
                predictions = model(data)
                loss = loss_fn(predictions, targets)
                loss = loss / CONFIG["accumulate_grad"]
        else:
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            loss = loss / CONFIG["accumulate_grad"]

        scaler.scale(loss).backward()

        if (i + 1) % CONFIG["accumulate_grad"] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        current_loss = loss.item() * CONFIG["accumulate_grad"]
        epoch_loss += current_loss
        loop.set_postfix(loss=current_loss)

    return epoch_loss / len(loader)


def main():
    print(f"Using Device: {CONFIG['device']}")
    torch.backends.cudnn.benchmark = True

    print("Initializing Datasets...")
    train_dataset = GeometryDataset(CONFIG["train_dir"], get_transforms("train"))
    val_dataset = GeometryDataset(CONFIG["val_dir"], get_transforms("val"))

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                              shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True, drop_last=True)

    # 1. 初始化空模型架构
    print("Initializing Model Architecture...")
    model = smp.Unet(
        encoder_name="tu-swin_large_patch4_window12_384_in22k",
        encoder_weights=None,  # 必须设为None，防止联网
        in_channels=3,
        classes=3,
        activation=None
    )

    # 2. 判断加载逻辑：是【续练】还是【新练】？
    resume_path = CONFIG["resume_checkpoint"]

    if resume_path and os.path.exists(resume_path):
        # === 逻辑 A: 续练 ===
        print("=" * 50)
        print(f"🚀 检测到存档，正在恢复: {resume_path}")
        print(f"🚀 将跳过前 {CONFIG['start_epoch']} 轮，从 Epoch {CONFIG['start_epoch'] + 1} 开始！")
        print("=" * 50)

        state_dict = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(state_dict)  # 加载完整模型参数
        print(">>> 历史存档加载成功。")

    else:
        # === 逻辑 B: 新练 (加载 backbone) ===
        print(f"Starting Fresh! Loading Backbone from: '{CONFIG['pretrained_path']}'")
        if os.path.exists(CONFIG["pretrained_path"]):
            state_dict = torch.load(CONFIG["pretrained_path"], map_location='cpu')
            target_model = model.encoder.model

            # 清洗 Key 名称
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("model.", "").replace("swin.", "")
                if "layers." in new_key: new_key = new_key.replace("layers.", "layers_")
                new_state_dict[new_key] = v

            try:
                target_model.load_state_dict(new_state_dict, strict=False)
                print(">>> Backbone weights loaded successfully.")
            except Exception as e:
                print(f">>> Loading backbone failed: {e}")
        else:
            print("[Warning] 没找到权重文件，将使用随机初始化！")

    model.to(CONFIG["device"])

    # 3. 优化器与调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-2)
    steps_per_epoch = len(train_loader) // CONFIG["accumulate_grad"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=steps_per_epoch * 2, T_mult=2, eta_min=1e-6
    )

    # 4. Loss
    dice_loss = smp.losses.DiceLoss(mode='multilabel')
    bce_loss = nn.BCEWithLogitsLoss()

    def criterion(pred, target):
        return 0.5 * dice_loss(pred, target) + 0.5 * bce_loss(pred, target)

    if torch.cuda.is_available():
        scaler = GradScaler('cuda')
    else:
        scaler = GradScaler()

    print("\nResuming Training Loop...")
    # 循环从 start_epoch 开始
    for epoch in range(CONFIG["start_epoch"], CONFIG["epochs"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        loss = train_fn(train_loader, model, optimizer, scaler, scheduler, criterion)
        print(f"Average Loss: {loss:.4f}")

        # 覆盖保存
        torch.save(model.state_dict(), "last_model.pth")
        print(">>> Model Saved: last_model.pth")


if __name__ == "__main__":
    main()