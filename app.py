import io
import base64
import yaml
import torch
import cv2
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import torchvision.transforms as transforms
from visualization.visualize_predictions import decode_segmap

# --- DINOv2 Model Definition (Matches our 10-epoch training) ---
class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.GELU()
        )
        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)

app = FastAPI(title="Offroad Segmentation AI")

# Mount the static frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

config = None
device = None
backbone = None
classifier = None
img_transform = None

@app.on_event("startup")
def load_model():
    global config, device, backbone, classifier, img_transform
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Server started. Loading model on {device}...")
    
    w = int(((960 / 4) // 14) * 14)
    h = int(((540 / 4) // 14) * 14)
    
    img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load local DINOv2
    repo_dir = os.path.abspath("dinov2")
    backbone = torch.hub.load(repo_or_dir=repo_dir, source='local', model='dinov2_vits14')
    backbone.to(device)
    backbone.eval()

    classifier = SegmentationHeadConvNeXt(384, 10, w//14, h//14)
    weights_path = "outputs/best_segmentation_head.pth"
    if os.path.exists(weights_path):
        classifier.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"SUCCESS: Loaded weights from {weights_path}")
    
    classifier.to(device)
    classifier.eval()

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("frontend/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hp, wp = image_rgb.shape[:2]
    
    input_tensor = img_transform(image_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = backbone.forward_features(input_tensor)["x_norm_patchtokens"]
        logits = classifier(features)
        outputs = F.interpolate(logits, size=(hp, wp), mode="bilinear", align_corners=False)
        pred_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
        
    color_pred = decode_segmap(pred_mask)
    
    def encode_base64_img(img_array, is_rgb=False):
        if is_rgb: img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', img_array)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

    return {
        "original_image": encode_base64_img(image_bgr),
        "segmented_image": encode_base64_img(color_pred, is_rgb=True)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
