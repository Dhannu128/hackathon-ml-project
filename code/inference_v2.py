import os
import cv2
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

# --- 1. CONFIGURATION ---
CFG = {
    "img_height": 512,
    "img_width": 896,
    "encoder": "efficientnet-b4",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": "best_unet_scse_finetuned.pth", # Naya fine-tuned model
    "test_dir": "terra-seg-dataset/offroad-seg-kaggle/test_images_padded"
}

def rle_encode(mask):
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

# --- 2. TRANSFORMS ---
transforms = A.Compose([
    A.Resize(CFG["img_height"], CFG["img_width"]),
    A.Normalize(),
    ToTensorV2()
])

# --- 3. LOAD MODEL ---
model = smp.UnetPlusPlus(
    encoder_name=CFG["encoder"], 
    encoder_weights=None, 
    in_channels=3, 
    classes=1, 
    decoder_attention_type="scse"
).to(CFG["device"])

# Load fine-tuned weights
if os.path.exists(CFG["model_path"]):
    print(f"✅ Loading Fine-tuned weights: {CFG['model_path']}")
    model.load_state_dict(torch.load(CFG["model_path"], map_location=CFG["device"]))
else:
    print("❌ Model file not found! Check the path.")
    exit()

model.eval()

# --- 4. INFERENCE LOOP WITH TTA ---
test_files = sorted(glob.glob(os.path.join(CFG['test_dir'], "*.png")) + 
                    glob.glob(os.path.join(CFG['test_dir'], "*.jpg")))
results = []

print("🚀 Starting Inference with TTA...")

with torch.no_grad():
    for img_path in tqdm(test_files):
        # Image read and preprocess
        orig_image = cv2.imread(img_path)
        if orig_image is None: continue
        orig_h, orig_w = orig_image.shape[:2]
        image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        
        # Normal prediction
        input_tensor = transforms(image=image_rgb)['image'].unsqueeze(0).to(CFG["device"])
        output_normal = model(input_tensor)
        
        # TTA: Horizontal Flip prediction
        input_flip = torch.flip(input_tensor, [3]) # Left-Right flip
        output_flip = model(input_flip)
        output_flip = torch.flip(output_flip, [3]) # Flip back
        
        # Combine (Average)
        final_output = (output_normal + output_flip) / 2.0
        
        # Resize back to original image size
        pred_resized = F.interpolate(final_output, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        
        # Apply Threshold
        mask = (torch.sigmoid(pred_resized) > 0.5).cpu().numpy().astype(np.uint8)[0, 0]

        # Prepare for CSV
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        results.append({"id": img_id, "rle_mask": rle_encode(mask)})

# --- 5. SAVE SUBMISSION ---
df = pd.DataFrame(results)
df.to_csv("submission_finetuned_v2.csv", index=False)
print(f"🎉 Done! Submission saved as: submission_finetuned_v2.csv")
