# predict_final_fruit_and_ripeness_CLEAN.py
# Put all photos of one fruit in "Input images" → Get perfect result

import torch
from torchvision import transforms, models
from PIL import Image
import os
from collections import Counter

# ------------------- PATHS -------------------
RIPENESS_MODEL = r"D:\programs\python\fruit reeping prediction\trained file\ripeness_model.pth"
FRUIT_MODEL    = r"D:\programs\python\fruit reeping prediction\trained file\fruit_type_ripe_only.pth"
INPUT_FOLDER   = r"D:\programs\python\fruit reeping prediction\Input images"

RIPENESS_CLASSES = ['Unripe', 'Ripe', 'Overripe']
FRUIT_CLASSES = ['apple', 'banana', 'guava', 'lime', 'mango',
                 'orange', 'pomegranate', 'strawberry', 'tomato']

# ------------------- PREPROCESS -------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- LOAD MODELS SAFELY (NO WARNINGS) -------------------
print("Loading models safely...")

model_rip = models.efficientnet_b3(weights=None)
model_rip.classifier[1] = torch.nn.Linear(model_rip.classifier[1].in_features, 3)
model_rip.load_state_dict(torch.load(RIPENESS_MODEL, map_location=device, weights_only=True))
model_rip.to(device)
model_rip.eval()

model_fruit = models.efficientnet_b3(weights=None)
model_fruit.classifier[1] = torch.nn.Linear(model_fruit.classifier[1].in_features, 9)
model_fruit.load_state_dict(torch.load(FRUIT_MODEL, map_location=device, weights_only=True))
model_fruit.to(device)
model_fruit.eval()

print("Models loaded successfully!\n" + "="*50)

# ------------------- ANALYZE ALL IMAGES -------------------
def analyze():
    files = [f for f in os.listdir(INPUT_FOLDER) 
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
    
    if not files:
        print("No images found in 'Input images' folder!")
        return

    print(f"Analyzing {len(files)} images...\n")

    ripeness_votes = []
    fruit_votes = []

    for f in files:
        path = os.path.join(INPUT_FOLDER, f)
        try:
            img = Image.open(path).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                ripeness_pred = model_rip(x).argmax(1).item()
                fruit_pred = model_fruit(x).argmax(1).item()
                ripeness_votes.append(ripeness_pred)
                fruit_votes.append(fruit_pred)
        except:
            continue

    total = len(ripeness_votes)
    if total == 0:
        print("All images failed to process!")
        return

    # Final fruit
    final_fruit = FRUIT_CLASSES[Counter(fruit_votes).most_common(1)[0][0]].upper()

    # Ripeness percentages
    unripe_pct  = ripeness_votes.count(0) / total * 100
    ripe_pct    = ripeness_votes.count(1) / total * 100
    overripe_pct = ripeness_votes.count(2) / total * 100

    # Bar function
    def bar(pct):
        return "█" * int(pct // 10) + "□" * (10 - int(pct // 10))

    # FINAL CLEAN OUTPUT
    print("FINAL RESULT")
    print("-" * 35)
    print(f"fruit     : {final_fruit}")
    print(f"unrip     : ({unripe_pct:5.1f}%)")
    print(f"rip       : ({ripe_pct:5.1f}%)")
    print(f"overrip   : ({overripe_pct:5.1f}%)")
    print("-" * 35)
    print(f"Based on {total} images")

# ------------------- RUN -------------------
if __name__ == "__main__":
    analyze()