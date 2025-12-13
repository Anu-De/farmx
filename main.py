import os
import pandas as pd
import re
import difflib
from datetime import datetime
import torch
from torchvision import transforms, models
from PIL import Image
from collections import Counter

from date_time_module import get_current_datetime
from map_module import get_coordinates_region_area
from weather_module import get_weather_data, get_season_from_weather


# =============== MODEL PATHS ==================
RIPENESS_MODEL = r"D:\programs\python\fruit reeping prediction\trained file\ripeness_model.pth"
FRUIT_MODEL    = r"D:\programs\python\fruit reeping prediction\trained file\fruit_type_ripe_only.pth"
INPUT_FOLDER   = r"D:\programs\python\fruit reeping prediction\Input images"

RIPENESS_CLASSES = ['Unripe', 'Ripe', 'Overripe']
FRUIT_CLASSES    = ['apple','banana','guava','lime','mango','orange','pomegranate','strawberry','tomato']


# =============== MODEL SETUP ==================
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RIPENESS MODEL
model_rip = models.efficientnet_b3(weights=None)
model_rip.classifier[1] = torch.nn.Linear(model_rip.classifier[1].in_features, 3)
model_rip.load_state_dict(torch.load(RIPENESS_MODEL, map_location=device, weights_only=True))
model_rip.to(device)
model_rip.eval()

# FRUIT MODEL
model_fruit = models.efficientnet_b3(weights=None)
model_fruit.classifier[1] = torch.nn.Linear(model_fruit.classifier[1].in_features, 9)
model_fruit.load_state_dict(torch.load(FRUIT_MODEL, map_location=device, weights_only=True))
model_fruit.to(device)
model_fruit.eval()


# ================= IMAGE PREDICTION ==================
def run_image_prediction():
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(
        ('.jpg','.jpeg','.png','.bmp','.webp'))]

    ripeness_votes = []
    fruit_votes = []

    for f in files:
        img = Image.open(os.path.join(INPUT_FOLDER, f)).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            rip = model_rip(x).argmax(1).item()
            fruit = model_fruit(x).argmax(1).item()

        ripeness_votes.append(rip)
        fruit_votes.append(fruit)

    fruit_name = FRUIT_CLASSES[Counter(fruit_votes).most_common(1)[0][0]].upper()

    unripe_pct     = ripeness_votes.count(0) / len(ripeness_votes) * 100
    ripe_pct       = ripeness_votes.count(1) / len(ripeness_votes) * 100
    overripe_pct   = ripeness_votes.count(2) / len(ripeness_votes) * 100

    if overripe_pct >= 50:
        ml_status = "OVERRIPE"
    elif ripe_pct >= 50:
        ml_status = "RIPE"
    else:
        ml_status = "UNRIPE"

    return fruit_name, ml_status, unripe_pct, ripe_pct, overripe_pct


# ============== LOGIC RIPENESS CALCULATION =============
months = ["January","February","March","April","May","June","July","August",
          "September","October","November","December"]

def month_to_number(month):
    try:
        return months.index(month.capitalize()) + 1
    except:
        return None

def calculate_month_diff(start, end):
    diff = end - start
    if diff < 0: diff += 12
    return diff

def calculate_ripeness(current_month, plantation_month):
    if plantation_month:
        crop_age = calculate_month_diff(plantation_month, current_month)
    else:
        crop_age = 6

    ripeness_score = min(100, max(10, crop_age * 12))

    if ripeness_score >= 80:
        phase_reason = "Harvest window period (Peak ripeness)"
    elif ripeness_score >= 50:
        phase_reason = "Inside ripening season"
    else:
        phase_reason = "Pre-ripening / vegetative growth phase"

    return ripeness_score, crop_age, phase_reason


# ===================== MAIN ==========================
def main():

    # Now ask user
    place = input("Enter Location / City / Village Name: ")
    plant_month = input("Enter Plantation Month (example: March): ")

    plantation_month_num = month_to_number(plant_month)

    # Current date
    now = datetime.strptime(get_current_datetime()['date'], "%d-%m-%Y")
    current_month = now.month
    month_name = now.strftime("%B")

    # Calculate crop age
    if plantation_month_num:
        crop_age_months = calculate_month_diff(plantation_month_num, current_month)
    else:
        crop_age_months = 6

    # ========= REQUIRED UPDATE (YOUR REQUEST) ==========
    if crop_age_months <= 2.5:
        print("\n==============================================")
        print("🌱 EARLY GROWTH STAGE DETECTED")
        print("==============================================")
        print(f"🪴 Plantation Month : {plant_month}")
        print(f"📆 Current Month     : {month_name}")
        print(f"🧮 Crop Age         : {crop_age_months:.1f} months")
        print("----------------------------------------------")
        print("🌱 STATUS: Plant is in GROWING PLANT STAGE.")
        print("❌ Image analysis skipped (too early for ripeness).")
        print("==============================================\n")
        return
    # ===================================================

    # FIRST — ML IMAGE PREDICTION
    print("\n🔍 Processing Image Prediction...\n")
    fruit_name, ml_status, unripe_pct, ripe_pct, overripe_pct = run_image_prediction()

    print(f"🍎 Fruit Detected : {fruit_name}\n")

    # CSV Lookup
    df = pd.read_csv(r"D:\programs\python\fruit reeping prediction\csv dataset\India_Fruit_Dataset.csv")
    result = df[df["Fruit"].str.lower().str.contains(fruit_name.lower())]

    rec_plant  = result["Plantation_Time"].iloc[0]
    rip_time   = result["Ripening_Time"].iloc[0]
    harv_time  = result["Harvesting_Time"].iloc[0]

    loc = get_coordinates_region_area(place)
    lat, lon = loc["coordinates"]
    district, state, country = loc["district"], loc["state"], loc["country"]

    weather = get_weather_data(place, state)
    predicted_season = get_season_from_weather(weather)

    ripeness_score, crop_age, phase_reason = calculate_ripeness(
        current_month, plantation_month_num
    )

    # ================= LOGIC RESULT =================
    print("\n==============================================")
    print("📄 RIPENESS PREDICTION REPORT (LOGIC)")
    print("==============================================")
    print(f"📍 Location Entered : {place}")
    print(f"📌 Fruit Selected   : {fruit_name}")
    print(f"🌐 Coordinates      : Lat={lat}, Lon={lon}")
    print(f"🏙 District         : {district}")
    print(f"🛣 State            : {state}")
    print(f"🌍 Country          : {country}")
    print("----------------------------------------------")
    print(f"🪴 Plantation Month (User)      : {plant_month}")
    print(f"🧾 Recommended Plantation Time  : {rec_plant}")
    print(f"📆 Current Month                : {month_name}")
    print(f"🍃 Standard Ripening Season     : {rip_time}")
    print(f"🌱 Standard Harvest Window      : {harv_time}")
    print(f"🌤 Weather Predicted Season     : {predicted_season}")
    print(f"📦 Crop Age                     : {crop_age} months of expected cycle")
    print("----------------------------------------------")
    print(f"📊 Ripeness Score               : {ripeness_score}%")
    print("----------------------------------------------")
    print(f"ℹ Reason                       : {phase_reason}")

    if ripeness_score >= 80:
        print("✅ STATUS: READY FOR HARVEST")
    elif ripeness_score >= 50:
        print("🟡 STATUS: Ripening / Near Peak")
    else:
        print("❌ STATUS: Unripe / Pre-ripening cycle")
    print("==============================================\n")


    # ================= ML RIPENESS =================
    print("==============================================")
    print("🖼 IMAGE RIPENESS PREDICTION REPORT (ML Output)")
    print("==============================================")
    print(f"🍎 Fruit Detected              : {fruit_name}")
    print(f"🟢 Unripe Percentage           : {unripe_pct:.1f}%")
    print(f"🟡 Ripe Percentage             : {ripe_pct:.1f}%")
    print(f"🔴 Overripe Percentage         : {overripe_pct:.1f}%")
    print(f"📊 ML Predicted Stage          : {ml_status}")
    print("==============================================\n")


    # ================= FINAL COMBINED RESULT =================
    print("=============== FINAL COMBINED OUTPUT ===============")
    print(f"📦 ML Ripeness Prediction  : {ml_status} ({ripe_pct:.1f}%)")
    print(f"🧠 Logic Ripeness Score     : {ripeness_score}%")
    print("-----------------------------------------------------")

    if ml_status == "OVERRIPE" or ripeness_score >= 90:
        result = "OVERRIPE / HARVEST IMMEDIATELY"
    elif ml_status == "RIPE" or ripeness_score >= 60:
        result = "RIPE / READY SOON"
    else:
        result = "UNRIPE / NOT READY"

    print(f"🏁 FINAL RESULT              : {result}")
    print("======================================================\n")


if __name__ == "__main__":
    main()
