import os
import pandas as pd
import re
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
CSV_PATH = r"D:\programs\python\fruit reeping prediction\csv dataset\India_Fruit_Dataset.csv"

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

    if not files:
        return "NO_FRUIT", "UNRIPE", 0, 0, 0

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


# ============== LOGIC HELPERS =================
months = ["January","February","March","April","May","June","July","August",
          "September","October","November","December"]

abbr_to_full = {
    'Jan': 'January', 'Feb': 'February', 'Mar': 'March', 'Apr': 'April',
    'May': 'May', 'Jun': 'June', 'Jul': 'July', 'Aug': 'August',
    'Sep': 'September', 'Oct': 'October', 'Nov': 'November', 'Dec': 'December'
}

def month_to_number(month):
    m = month.capitalize().strip()
    if m in abbr_to_full:
        m = abbr_to_full[m]
    try:
        return months.index(m) + 1
    except ValueError:
        return None

def calculate_month_diff(start, end):
    diff = end - start
    if diff < 0: diff += 12
    return diff

def ripening_month_list(r):
    r = r.replace('–', '-')
    parts = r.split('-')
    if len(parts) != 2:
        return []
    a, b = [p.strip() for p in parts]
    num_a = month_to_number(a)
    num_b = month_to_number(b)
    if num_a is None or num_b is None:
        return []
    # Fix for year-wrap (e.g., Dec-Jan)
    if num_a > num_b:
        return list(range(num_a, 13)) + list(range(1, num_b + 1))
    else:
        return list(range(num_a, num_b + 1))

def parse_crop_cycle(ripening_time):
    ripening_time = ripening_time.replace('–', '-')
    if 'Months' in ripening_time:
        nums = re.findall(r'\d+', ripening_time)
        if len(nums) >= 2:
            return int((int(nums[0]) + int(nums[1])) / 2)
        elif len(nums) == 1:
            return int(nums[0])
        else:
            return 12
    else:
        return 12

def is_ripening_duration(ripening_time):
    return 'Months' in ripening_time

def get_ripening_duration_range(ripening_time):
    ripening_time = ripening_time.replace('–', '-')
    nums = re.findall(r'\d+', ripening_time)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    elif len(nums) == 1:
        return int(nums[0]), int(nums[0])
    else:
        return None, None

def is_all_year_season(csv_season):
    return 'all year' in csv_season.lower() or 'year-round' in csv_season.lower()


# ================= UPDATED LOGIC SCORE =================
def logic_score(plant_month, curr_month, crop_cycle,
                ripening_time, csv_season, csv_region,
                detected_season, detected_region, crop_age):

    if crop_age <= 2.5:
        return 0, crop_age, "Growing stage (too early)"

    # Base score (age vs cycle)
    base = min((crop_age / crop_cycle) * 50, 50)

    # All-year check
    all_year = is_all_year_season(csv_season)

    # Ripening match
    rip_score = 5  # Default
    if all_year:
        if is_ripening_duration(ripening_time):
            min_mo, max_mo = get_ripening_duration_range(ripening_time)
            if min_mo is not None:
                if min_mo <= crop_age <= max_mo:
                    rip_score = 20
                elif crop_age == min_mo - 1:
                    rip_score = 10
        else:
            rip_score = 10
    else:
        if is_ripening_duration(ripening_time):
            rip_score = 10
        else:
            if curr_month in ripening_month_list(ripening_time):
                rip_score = 20

    # Season match
    season_score = 15 if all_year else (15 if detected_season.lower() in csv_season.lower() else 5)

    # Location match
    loc_score = 15 if detected_region.lower() in csv_region.lower() else 5

    final = min(base + rip_score + season_score + loc_score, 100)

    return final, crop_age, "Logic-based ripeness score"


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

    # CSV Lookup with state preference
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print("CSV file not found! Using defaults.")
        return

    loc = get_coordinates_region_area(place)
    lat, lon = loc["coordinates"]
    district, state, country = loc["district"], loc["state"], loc["country"]

    weather = get_weather_data(place, state)
    predicted_season = get_season_from_weather(weather)

    # Filter by fruit first
    fruit_matches = df[df["Fruit"].str.lower().str.contains(fruit_name.lower())]

    # Prefer state match
    state_lower = state.lower()
    state_match = fruit_matches[fruit_matches["State"].str.lower().str.contains(state_lower, na=False)]
    if not state_match.empty:
        result = state_match.iloc[0]
    else:
        result = fruit_matches.iloc[0] if not fruit_matches.empty else pd.Series()  # Default empty

    if result.empty:
        print("No CSV data for fruit! Skipping logic.")
        return

    rec_plant  = result["Plantation_Time"]
    rip_time   = result["Ripening_Time"]
    harv_time  = result["Harvesting_Time"]
    crop_cycle = parse_crop_cycle(rip_time)
    csv_season = result["Season"]
    csv_region = result["State"]

    score, crop_age, phase_reason = logic_score(
        plantation_month_num,
        current_month,
        crop_cycle,
        rip_time,
        csv_season,
        csv_region,
        predicted_season,
        state,
        crop_age_months
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
    print(f"🔄 Crop Cycle (from CSV)        : {crop_cycle} months")
    print("----------------------------------------------")
    print(f"📊 Ripeness Score               : {score:.1f}%")
    print("----------------------------------------------")
    print(f"ℹ Reason                       : {phase_reason}")

    if score >= 80:
        print("✅ STATUS: READY FOR HARVEST")
    elif score >= 50:
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
    print(f"🧠 Logic Ripeness Score     : {score:.1f}%")
    print("-----------------------------------------------------")

    # Blended Combined Score (70% ML + 30% Logic)
    combined_score = min(100, (0.7 * ripe_pct) + (0.3 * score))

    if combined_score >= 90:
        result = "OVERRIPE / HARVEST IMMEDIATELY"
    elif combined_score >= 70:
        result = "RIPE / READY SOON"
    elif combined_score >= 40:
        result = "NEAR RIPE / WAIT SOME TIME"
    else:
        result = "NOT RIPE / GROW MORE"

    print(f"🔗 Combined Score (70% ML + 30% Logic): {combined_score:.1f}%")
    print(f"🏁 FINAL RESULT              : {result}")
    print("======================================================\n")


if __name__ == "__main__":
    main()
