import cv2
import numpy as np
import random

def degrade_image(img):
#     img = img.astype(np.float32)/255.0

#     #Negative Inversion
#     img = 1.0 - img

#     #Color Fading
#     fade = np.random.uniform(0.6,0.95)
#     img *= fade

#     #Gaussian Noise
#     noise = np.random.normal(0, 0.03, img.shape).astype(np.float32)
#     img = img + noise

#     #Blur
#     img = cv2.GaussianBlur(img,(5,5),0)

#     return np.clip(img*255,0,255).astype(np.uint8)

#ADVANCED PIPELINE
    # Normalize
    img = img.astype(np.float32) / 255.0

    # 1. Negative inversion
    img = 1.0 - img

    # 2. Channel-wise color imbalance (film dye simulation)
    channel_scale = np.random.uniform(
        [0.9, 0.6, 0.4],   # R, G, B lower bounds
        [1.2, 1.0, 0.8],   # R, G, B upper bounds
        size=(1,1,3)
    )
    img = img * channel_scale

    # 3. Cross-channel color leakage (film crosstalk)
    mix = np.random.uniform(0.02, 0.08)
    img[:,:,1] += mix * img[:,:,0]   # R -> G
    img[:,:,2] += mix * img[:,:,1]   # G -> B

    # 4. Exposure nonlinearity (gamma variation)
    gamma = np.random.uniform(0.7, 1.3)
    img = np.power(np.clip(img, 0, 1), gamma)

    # 5. Signal-dependent film grain
    noise_strength = np.random.uniform(0.01, 0.04)
    noise = np.random.normal(
        0, noise_strength, img.shape
    ).astype(np.float32)
    img = img + noise * img

    # 6. Slight blur (scanner + film softness)
    if random.random() < 0.5:
        img = cv2.GaussianBlur(img, (3,3), 0)

    # Final clamp
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)
