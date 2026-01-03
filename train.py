import torch
import cv2
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.cnn_restorer import ResNetRestoration
from degradation.degrade import degrade_image


class FilmDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.files = os.listdir(img_dir)
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.files[idx])
        clean = cv2.imread(path)
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)
        clean = cv2.resize(clean, (512,512))

        degraded = degrade_image(clean.copy())

        clean = self.transform(clean)
        degraded = self.transform(degraded)

        return degraded, clean


def train():
    dataset = FilmDataset("sample_images")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = ResNetRestoration().cuda()
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    for epoch in range(10):
        total_loss = 0
        for degraded, clean in loader:
            degraded = degraded.cuda()
            clean = clean.cuda()

            optimizer.zero_grad()
            output = model(degraded)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss = {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
    train()
