import torch
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from models.cnn_restorer import ResNetRestoration

def evaluate(clean_path, degraded_path):
    clean = cv2.imread(clean_path)
    degraded = cv2.imread(degraded_path)

    clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
    degraded = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)

    clean = cv2.resize(clean, (512,512))
    degraded = cv2.resize(degraded, (512,512))

    clean_norm = clean.astype(np.float32) / 255.0
    degraded_norm = degraded.astype(np.float32) / 255.0

    model = ResNetRestoration().cuda()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    inp = torch.from_numpy(degraded_norm).permute(2,0,1).unsqueeze(0).cuda()

    with torch.no_grad():
        restored = model(inp).cpu().squeeze(0).permute(1,2,0).numpy()

    psnr = peak_signal_noise_ratio(clean_norm, restored)
    ssim = structural_similarity(clean_norm, restored, channel_axis=2)

    print("PSNR:", psnr)
    print("SSIM:", ssim)

if __name__ == "__main__":
    evaluate("sample_images/clean.png","sample_images/degraded.png")
