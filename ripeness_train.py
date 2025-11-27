import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, classification_report
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm  # <-- This gives you the nice progress bar

# -----------------------------
# PATHS
# -----------------------------
TRAIN_DIR = r"D:\programs\python\fruit reeping prediction\image dataset\Train"
TEST_DIR  = r"D:\programs\python\fruit reeping prediction\image dataset\Test"
SAVE_PATH = r"D:\programs\python\fruit reeping prediction\trained file\ripeness_model.pth"

IMG_SIZE = 224
BATCH = 32
EPOCHS = 25
LR = 3e-4

# -----------------------------
# DATASET
# -----------------------------
class RipenessDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.class_map = {"Unripe": 0, "Ripe": 1, "Overripe": 2}
        self.class_names = ["Unripe", "Ripe", "Overripe"]

        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: {class_dir} not found!")
                continue
            label = self.class_map[class_name]
            for fruit_name in os.listdir(class_dir):
                fruit_dir = os.path.join(class_dir, fruit_name)
                if not os.path.isdir(fruit_dir):
                    continue
                for img_name in os.listdir(fruit_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(fruit_dir, img_name))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# -----------------------------
# TRANSFORMS
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -----------------------------
# MAIN - Windows Safe
# -----------------------------
if __name__ == '__main__':
    train_ds = RipenessDataset(TRAIN_DIR, train_transform)
    test_ds  = RipenessDataset(TEST_DIR, test_transform)

    print(f"Classes: {train_ds.class_names}")
    print(f"Train samples: {len(train_ds):,}")
    print(f"Test samples : {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

    model = models.efficientnet_b3(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("\n" + "="*70)
    print("STARTING TRAINING - FRUIT RIPENESS CLASSIFIER (3 classes)")
    print("="*70)

    best_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEPOCH {epoch}/{EPOCHS}  [{'='*(epoch*2):<50}]  LR: {optimizer.param_groups[0]['lr']:.2e}")

        # ---------------- Training ----------------
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Training  ", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = running_loss / len(train_loader)

        # ---------------- Validation ----------------
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="Validating", leave=False):
                inputs = inputs.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        scheduler.step()

        print(f"Epoch {epoch:2d} | Train Loss: {avg_train_loss:.4f} | Test Accuracy: {acc*100:6.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"NEW BEST MODEL! → {best_acc*100:.2f}% accuracy")

    print("\n" + "="*70)
    print(f"TRAINING FINISHED! BEST ACCURACY: {best_acc*100:.2f}%")
    print(f"Model saved: {SAVE_PATH}")
    print("\nFinal Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=train_ds.class_names))
    print("="*70)