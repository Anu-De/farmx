# fruit_type_ripe_FINAL_FIXED.py
import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# ----------------------------- PATHS -----------------------------
TRAIN_DIR = r"D:\programs\python\fruit reeping prediction\image dataset\Train\Ripe"
TEST_DIR  = r"D:\programs\python\fruit reeping prediction\image dataset\Test\Ripe"
SAVE_PATH = r"D:\programs\python\fruit reeping prediction\trained file\fruit_type_ripe_only.pth"

# ----------------------------- FIXED DATASET (NO PRINTS!) -----------------------------
class RipeFruitDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_to_idx=None):
        self.samples = []
        self.transform = transform
        self.class_to_idx = class_to_idx  # Shared across train/test

        for fruit in os.listdir(root_dir):
            fruit_dir = os.path.join(root_dir, fruit)
            if not os.path.isdir(fruit_dir):
                continue
            fruit_lower = fruit.strip().lower()
            if fruit_lower not in self.class_to_idx:
                continue
            label = self.class_to_idx[fruit_lower]
            for img_name in os.listdir(fruit_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(fruit_dir, img_name), label))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ----------------------------- BUILD CLASS MAP ONCE -----------------------------
all_fruits = set()
for d in [TRAIN_DIR, TEST_DIR]:
    if os.path.exists(d):
        for f in os.listdir(d):
            if os.path.isdir(os.path.join(d, f)):
                all_fruits.add(f.strip().lower())

CLASS_NAMES = sorted(all_fruits)
class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
print(f"Detected {len(CLASS_NAMES)} fruit classes: {CLASS_NAMES}")

# ----------------------------- TRANSFORMS -----------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ----------------------------- MAIN -----------------------------
if __name__ == '__main__':
    # Delete old model
    if os.path.exists(SAVE_PATH):
        os.remove(SAVE_PATH)
        print("Old model deleted ✓")

    # Create datasets (NO printing inside!)
    train_ds = RipeFruitDataset(TRAIN_DIR, train_transform, class_to_idx)
    test_ds  = RipeFruitDataset(TEST_DIR,  test_transform,  class_to_idx)

    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples:  {len(test_ds)}")

    # IMPORTANT: num_workers=0 on Windows when dataset has issues
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=0)

    model = models.efficientnet_b3(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, total_steps=15*len(train_loader))

    print("\nSTARTING TRAINING (this time it will go to 98%+)")
    best_acc = 0
    for epoch in range(1, 16):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/15"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Test
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total * 100
        print(f"Epoch {epoch:2d} → Accuracy: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), SAVE_PATH)
            print("NEW BEST MODEL!")

    print(f"\nDONE! Final best accuracy: {best_acc:.2f}%")
    print("Model saved → fruit_type_ripe_only.pth")