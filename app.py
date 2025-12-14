import streamlit as st
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from datetime import datetime
import re

# --- IMPORT CUSTOM MODULES ---
try:
    from map_module import get_coordinates_region_area
    from weather_module import get_weather_data, get_season_from_weather
except ImportError:
    def get_coordinates_region_area(loc): return {"district": "Unknown", "state": "Unknown", "country": "India", "coordinates": (0,0)}
    def get_weather_data(loc, st): return None
    def get_season_from_weather(w): return "Unknown"

# --- CONFIGURATION ---
st.set_page_config(page_title="FARMX - CROP RIPENESS PREDICTION SYSTEM", page_icon="🌱", layout="wide")

# --- CUSTOM CSS FOR MODERN UI ---
st.markdown("""
    <style>
    /* Make metrics stand out */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700;
    }
    /* Style the header */
    .main-header {
        font-size: 3rem; 
        color: #4CAF50; 
        text-align: center; 
        font-weight: bold;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- PATHS (Same as before) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RIPENESS_MODEL_PATH = os.path.join(BASE_DIR, "trained file", "ripeness_model.pth")
FRUIT_MODEL_PATH = os.path.join(BASE_DIR, "trained file", "fruit_type_ripe_only.pth")
CSV_PATH = os.path.join(BASE_DIR, "csv dataset", "India_Fruit_Dataset.csv")

RIPENESS_CLASSES = ['Unripe', 'Ripe', 'Overripe']
FRUIT_CLASSES    = ['apple','banana','guava','lime','mango','orange','pomegranate','strawberry','tomato']

# --- HELPER FUNCTIONS (Same as before) ---
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
        return 1

def calculate_month_diff(start, end):
    diff = end - start
    if diff < 0: diff += 12
    return diff

def ripening_month_list(r):
    if not isinstance(r, str): return []
    r = r.replace('–', '-')
    parts = r.split('-')
    if len(parts) != 2: return []
    a, b = [p.strip() for p in parts]
    num_a = month_to_number(a)
    num_b = month_to_number(b)
    if num_a > num_b:
        return list(range(num_a, 13)) + list(range(1, num_b + 1))
    else:
        return list(range(num_a, num_b + 1))

def parse_crop_cycle(ripening_time):
    if not isinstance(ripening_time, str): return 12
    ripening_time = ripening_time.replace('–', '-')
    if 'Months' in ripening_time:
        nums = re.findall(r'\d+', ripening_time)
        if len(nums) >= 2:
            return int((int(nums[0]) + int(nums[1])) / 2)
        elif len(nums) == 1:
            return int(nums[0])
    return 12

def is_ripening_duration(ripening_time):
    return 'Months' in str(ripening_time)

def get_ripening_duration_range(ripening_time):
    ripening_time = str(ripening_time).replace('–', '-')
    nums = re.findall(r'\d+', ripening_time)
    if len(nums) >= 2:
        return int(nums[0]), int(nums[1])
    elif len(nums) == 1:
        return int(nums[0]), int(nums[0])
    return None, None

def is_all_year_season(csv_season):
    return 'all year' in str(csv_season).lower() or 'year-round' in str(csv_season).lower()

def calculate_logic_score(plant_month_num, curr_month_num, crop_cycle, 
                          ripening_time, csv_season, csv_region, 
                          detected_season, detected_region, crop_age):
    
    base = min((crop_age / crop_cycle) * 50, 50)
    all_year = is_all_year_season(csv_season)
    rip_score = 5 
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
            if curr_month_num in ripening_month_list(ripening_time):
                rip_score = 20

    season_score = 15 if all_year else (15 if detected_season.lower() in str(csv_season).lower() else 5)
    loc_score = 15 if detected_region.lower() in str(csv_region).lower() else 5

    final = min(base + rip_score + season_score + loc_score, 100)
    return final, "Logic-based ripeness score"

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    device = torch.device("cpu") 
    
    model_rip = models.efficientnet_b3(weights=None)
    model_rip.classifier[1] = torch.nn.Linear(model_rip.classifier[1].in_features, 3)
    if os.path.exists(RIPENESS_MODEL_PATH):
        model_rip.load_state_dict(torch.load(RIPENESS_MODEL_PATH, map_location=device, weights_only=True))
    else:
        st.error("Ripeness Model not found.")
        return None, None, None
    model_rip.to(device)
    model_rip.eval()

    model_fruit = models.efficientnet_b3(weights=None)
    model_fruit.classifier[1] = torch.nn.Linear(model_fruit.classifier[1].in_features, 9)
    if os.path.exists(FRUIT_MODEL_PATH):
        model_fruit.load_state_dict(torch.load(FRUIT_MODEL_PATH, map_location=device, weights_only=True))
    else:
        st.error("Fruit Model not found.")
        return None, None, None
    model_fruit.to(device)
    model_fruit.eval()

    return model_rip, model_fruit, device

def get_prediction(image, model_rip, model_fruit, device):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    
    img_t = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        fruit_idx = model_fruit(img_t).argmax(1).item()
        fruit_name = FRUIT_CLASSES[fruit_idx].upper()

        rip_logits = model_rip(img_t)
        probs = F.softmax(rip_logits, dim=1)
        
        unripe_pct   = probs[0][0].item() * 100
        ripe_pct     = probs[0][1].item() * 100
        overripe_pct = probs[0][2].item() * 100
        
        if overripe_pct >= 50:
            ml_status = "OVERRIPE"
        elif ripe_pct >= 50:
            ml_status = "RIPE"
        else:
            ml_status = "UNRIPE"

    return fruit_name, ml_status, ripe_pct

# ================= MODERNIZED UI =================

# 1. Custom Header
st.markdown('<div class="main-header">🌱 FARMX - CROP RIPENESS PREDICTION SYSTEM</div>', unsafe_allow_html=True)

model_rip, model_fruit, device = load_models()

# 2. Field Parameters (Top Main)
st.write("### 1️⃣ Field Configuration")
with st.container(border=True):
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        location = st.text_input("📍 Location / City", "Pune, Maharashtra")
    with col_p2:
        months_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        plantation_month = st.selectbox("🗓️ Plantation Month", months_list, index=0)

# --- CALCULATE CROP AGE ---
now = datetime.now()
current_month_name = now.strftime("%B")
current_month_num = now.month
plant_month_num = month_to_number(plantation_month)
crop_age_months = calculate_month_diff(plant_month_num, current_month_num)

# --- EARLY GROWTH CHECK ---
if crop_age_months <= 2.5:
    st.divider()
    st.warning("🌱 EARLY GROWTH STAGE DETECTED")
    st.info(f"""
    **Status:** Plant is in Growing Stage.
    \n**Reason:** Crop age ({crop_age_months} months) is too low for ripeness analysis.
    \n**Action:** Image analysis is disabled to prevent false positives.
    """)
    st.stop()

# 3. Image Upload
st.write("### 2️⃣ Upload Crop Image")
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # --- SPLIT LAYOUT: Image (Left) vs Analysis (Right) ---
    col_img, col_report = st.columns([1, 1])
    
    with col_img:
        st.write("**Image Preview:**")
        st.image(image, caption='Uploaded Crop', use_column_width=True) 

    with col_report:
        st.write("**Action & Results:**")
        if st.button("🔍 Analyze Ripeness", type="primary", use_container_width=True):
            if model_rip:
                with st.spinner('Processing Logic + ML Models...'):
                    # --- ANALYSIS LOGIC ---
                    fruit_name, ml_status, ripe_pct = get_prediction(image, model_rip, model_fruit, device)
                    
                    loc_data = get_coordinates_region_area(location)
                    district = loc_data.get("district", "Unknown")
                    state = loc_data.get("state", "Unknown")
                    weather = get_weather_data(location, state)
                    predicted_season = get_season_from_weather(weather)

                    logic_score_val = 0
                    rec_plant = "Unknown"
                    rip_time = "Unknown"
                    harv_time = "Unknown"
                    
                    try:
                        df = pd.read_csv(CSV_PATH)
                        fruit_matches = df[df["Fruit"].str.lower().str.contains(fruit_name.lower())]
                        state_lower = state.lower()
                        state_match = fruit_matches[fruit_matches["State"].str.lower().str.contains(state_lower, na=False)]
                        
                        if not state_match.empty:
                            result = state_match.iloc[0]
                        else:
                            result = fruit_matches.iloc[0] if not fruit_matches.empty else pd.Series()

                        if not result.empty:
                            rec_plant  = result.get("Plantation_Time", "Unknown")
                            rip_time   = result.get("Ripening_Time", "Unknown")
                            harv_time  = result.get("Harvesting_Time", "Unknown")
                            csv_season = result.get("Season", "Unknown")
                            csv_region = result.get("State", "Unknown")
                            crop_cycle = parse_crop_cycle(rip_time)
                            
                            logic_score_val, _ = calculate_logic_score(
                                plant_month_num, current_month_num, crop_cycle,
                                rip_time, csv_season, csv_region,
                                predicted_season, state, crop_age_months
                            )
                    except Exception as e:
                        st.error(f"Error reading CSV data: {e}")

                    combined_score = min(100, (0.7 * ripe_pct) + (0.3 * logic_score_val))

                    if combined_score >= 90:
                        final_status = "OVERRIPE / HARVEST IMMEDIATELY"
                        status_color = "red"
                        alert_func = st.error
                    elif combined_score >= 70:
                        final_status = "RIPE / READY SOON"
                        status_color = "green"
                        alert_func = st.success
                    elif combined_score >= 40:
                        final_status = "NEAR RIPE / WAIT SOME TIME"
                        status_color = "orange"
                        alert_func = st.warning
                    else:
                        final_status = "NOT RIPE / GROW MORE"
                        status_color = "grey"
                        alert_func = st.info

                    # --- SIDE-BY-SIDE RESULT CARD ---
                    with st.container(border=True):
                        # Top Row: Metrics
                        m1, m2 = st.columns(2)
                        m1.metric("🍎 Fruit Detected", fruit_name)
                        m2.metric("📈 Ripeness Score", f"{combined_score:.1f}%")
                        
                        # Middle Row: Status Banner
                        st.write("") # Spacer
                        alert_func(f"**STATUS:** {final_status}")
                        st.progress(combined_score / 100)

                        # Bottom Row: Expander
                        with st.expander("See Technical Details"):
                            st.markdown(f"**🗓 Plantation Month:** {plantation_month}") # Added Requested Line
                            st.markdown(f"**📍 Location:** {district}, {state}")
                            st.markdown(f"**⏳ Crop Age:** {crop_age_months} months")
                            st.markdown(f"**🌤 Weather:** {predicted_season}")
                            
                            # REMOVED LOGIC SCORE AS REQUESTED
                            st.markdown(f"**📅 Standard Cycle:** {rip_time}")
                            st.markdown(f"**🚜 Harvest Window:** {harv_time}")
