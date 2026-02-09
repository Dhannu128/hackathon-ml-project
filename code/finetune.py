import os
import cv2
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast, GradScaler
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# --- 1. CONFIGURATION ---
CFG = {
    "img_height": 512,       # 32 se divisible hai (16 * 32)
    "img_width": 896,        # 32 se divisible hai (28 * 32)
    "batch_size": 4,          
    "accum_iter": 8,          
    "encoder": "efficientnet-b4", 
    "lr": 5e-5,               
    "epochs": 10,             
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "mandated_classes": [27, 39],
    "load_pretrained": True,  
    "model_path": "best_unet_scse.pth", 
    "train_img_dir": "terra-seg-dataset/offroad-seg-kaggle/train_images",
    "train_mask_dir": "terra-seg-dataset/offroad-seg-kaggle/train_masks",
}

# --- 2. DATASET ---
class TerraSegDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.float32)
        else:
            mask = np.isin(mask, CFG["mandated_classes"]).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        return image, mask.unsqueeze(0)

# --- 3. AUGMENTATIONS (FIXED: Strict Resize to 512x896) ---
transforms = A.Compose([
    # Pehle random crop/resize hoga
    A.RandomResizedCrop(
        size=(CFG["img_height"], CFG["img_width"]), 
        scale=(0.8, 1.0), 
        p=0.5
    ),
    # Phir ensure karenge ki final size wahi ho jo model chahta hai
    A.Resize(height=CFG["img_height"], width=CFG["img_width"]),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(p=0.5),
        A.CLAHE(clip_limit=4.0, p=0.5),
    ], p=0.4),
    A.Normalize(),
    ToTensorV2()
])

# --- 4. DATA LOADERS ---
train_imgs = sorted(glob.glob(os.path.join(CFG["train_img_dir"], "*.png")) + 
                    glob.glob(os.path.join(CFG["train_img_dir"], "*.jpg")))
train_masks = sorted(glob.glob(os.path.join(CFG["train_mask_dir"], "*.png")))

full_ds = TerraSegDataset(train_imgs, train_masks, transform=transforms)
train_size = int(0.90 * len(full_ds)) 
train_ds, val_ds = random_split(full_ds, [train_size, len(full_ds)-train_size])

train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

# --- 5. MODEL ---
model = smp.UnetPlusPlus(
    encoder_name=CFG["encoder"], 
    encoder_weights=None, 
    in_channels=3, 
    classes=1, 
    decoder_attention_type="scse"
)

if CFG["load_pretrained"] and os.path.exists(CFG["model_path"]):
    print(f"✅ Loading model weights from: {CFG['model_path']}")
    state_dict = torch.load(CFG["model_path"], map_location=CFG["device"])
    model.load_state_dict(state_dict)

model = model.to(CFG["device"])

# --- 6. LOSS ---
dice_loss = smp.losses.DiceLoss(mode='binary', smooth=1.0)
focal_loss = smp.losses.FocalLoss(mode='binary', gamma=2.0)

def criterion(pred, mask):
    return 0.7 * dice_loss(pred, mask) + 0.3 * focal_loss(pred, mask)

# --- 7. OPTIMIZER & SCHEDULER ---
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-7)

# Fixed for Torch 2.4+
scaler = GradScaler('cuda')

best_iou = 0.7788 

# --- 8. TRAINING LOOP ---
for epoch in range(CFG["epochs"]):
    model.train()
    total_loss = 0
    optimizer.zero_grad() 
    
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG['epochs']}")
    
    for i, (imgs, masks) in enumerate(loop):
        imgs, masks = imgs.to(CFG["device"]), masks.to(CFG["device"])
        
        with autocast('cuda'):
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss = loss / CFG["accum_iter"]
            
        scaler.scale(loss).backward()
        
        if ((i + 1) % CFG["accum_iter"] == 0) or (i + 1 == len(train_loader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step(epoch + i / len(train_loader)) 
        
        loop.set_postfix(loss=f"{loss.item()*CFG['accum_iter']:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    # Validation
    model.eval()
    val_iou = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(CFG["device"]), masks.to(CFG["device"])
            outputs = model(imgs)
            
            # Binary Preds
            preds = (torch.sigmoid(outputs) > 0.5).float()
            intersection = (preds * masks).sum()
            union = preds.sum() + masks.sum() - intersection
            val_iou += (intersection + 1e-6) / (union + 1e-6)
            
    avg_iou = (val_iou / len(val_loader)).item()
    print(f"Epoch {epoch+1} | Val IoU: {avg_iou:.4f}")
    
    if avg_iou > best_iou:
        best_iou = avg_iou
        torch.save(model.state_dict(), "best_unet_scse_finetuned.pth")
        print(f"    🌟 New Best Saved: {best_iou:.4f}")

print("🏁 Finished!")
