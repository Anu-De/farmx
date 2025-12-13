import streamlit as st
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from datetime import datetime

# --- IMPORT CUSTOM MODULES ---
try:
    from map_module import get_coordinates_region_area
    from weather_module import get_weather_data, get_season_from_weather
except ImportError:
    # Fallback if modules are missing
    def get_coordinates_region_area(loc): return {"district": "Unknown", "state": "Unknown"}
    def get_weather_data(loc, st): return None
    def get_season_from_weather(w): return "Unknown"

# --- CONFIGURATION ---
st.set_page_config(page_title="FarmX - Crop Ripeness", page_icon="🌱", layout="wide")

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RIPENESS_MODEL_PATH = os.path.join(BASE_DIR, "trained file", "ripeness_model.pth")
FRUIT_MODEL_PATH = os.path.join(BASE_DIR, "trained file", "fruit_type_ripe_only.pth")
CSV_PATH = os.path.join(BASE_DIR, "csv dataset", "India_Fruit_Dataset.csv")

RIPENESS_CLASSES = ['Unripe', 'Ripe', 'Overripe']
FRUIT_CLASSES    = ['apple','banana','guava','lime','mango','orange','pomegranate','strawberry','tomato']

# --- LOAD MODELS (CPU MODE) ---
@st.cache_resource
def load_models():
    device = torch.device("cpu") 
    
    # Load Ripeness Model
    model_rip = models.efficientnet_b3(weights=None)
    model_rip.classifier[1] = torch.nn.Linear(model_rip.classifier[1].in_features, 3)
    if os.path.exists(RIPENESS_MODEL_PATH):
        model_rip.load_state_dict(torch.load(RIPENESS_MODEL_PATH, map_location=device, weights_only=True))
    else:
        st.error("Ripeness Model not found.")
        return None, None, None
    model_rip.to(device)
    model_rip.eval()

    # Load Fruit Model
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

# --- PREDICTION LOGIC ---
def get_prediction(image, model_rip, model_fruit, device):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    
    img_t = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # 1. Get Fruit Type
        fruit_idx = model_fruit(img_t).argmax(1).item()
        fruit_name = FRUIT_CLASSES[fruit_idx].upper()

        # 2. Get Ripeness Probabilities (Softmax)
        rip_logits = model_rip(img_t)
        probs = F.softmax(rip_logits, dim=1)  # Converts logits to percentages (0.0 - 1.0)
        
        # EXTRACT SPECIFIC PERCENTAGES (Just like main.py but using probability)
        # Class 0 = Unripe, Class 1 = Ripe, Class 2 = Overripe
        unripe_pct   = probs[0][0].item() * 100
        ripe_pct     = probs[0][1].item() * 100  # <--- This is the one you asked for
        overripe_pct = probs[0][2].item() * 100
        
        # Determine Status
        if overripe_pct >= 50:
            ml_status = "OVERRIPE"
        elif ripe_pct >= 50:
            ml_status = "RIPE"
        else:
            ml_status = "UNRIPE"

    return fruit_name, ml_status, ripe_pct

def calculate_logic_ripeness(current_month_name, plantation_month_name):
    months = ["January","February","March","April","May","June","July","August",
              "September","October","November","December"]
    
    try:
        plant_month_num = months.index(plantation_month_name) + 1
        curr_month_num = months.index(current_month_name) + 1
    except:
        return 0, 0, "Error in dates"

    def calculate_month_diff(start, end):
        diff = end - start
        if diff < 0: diff += 12
        return diff

    crop_age = calculate_month_diff(plant_month_num, curr_month_num)
    ripeness_score = min(100, max(10, crop_age * 12))

    if ripeness_score >= 80:
        phase_reason = "Harvest window period"
    elif ripeness_score >= 50:
        phase_reason = "Inside ripening season"
    else:
        phase_reason = "Vegetative growth phase"

    return ripeness_score, crop_age, phase_reason

# ================= MAIN UI =================
st.title("🌱 FarmX: Crop Ripeness System")

model_rip, model_fruit, device = load_models()

# --- SIDEBAR ---
st.sidebar.header("1. Field Parameters")
location = st.sidebar.text_input("Location / City", "Pune, Maharashtra")

months_list = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
plantation_month = st.sidebar.selectbox("Plantation Month", months_list, index=8)
current_month_name = datetime.now().strftime("%B")

# --- MAIN INTERFACE ---
st.header("2. Upload Crop Image")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with col2:
        st.subheader("Analysis Results")
        
        if st.button("Analyze Ripeness"):
            if model_rip:
                with st.spinner('Analyzing...'):
                    # 1. AI Prediction
                    fruit_name, ml_status, ripe_pct = get_prediction(image, model_rip, model_fruit, device)
                    
                    # 2. Logic Data
                    loc_data = get_coordinates_region_area(location)
                    district = loc_data.get("district", "Unknown")
                    state = loc_data.get("state", "Unknown")
                    
                    weather = get_weather_data(location, state)
                    predicted_season = get_season_from_weather(weather)

                    logic_score, crop_age, phase_reason = calculate_logic_ripeness(current_month_name, plantation_month)
                    
                    # --- DISPLAY RESULTS ---
                    
                    m1, m2, m3 = st.columns(3)
                    
                    m1.metric("Fruit", fruit_name)
                    m2.metric("Stage", ml_status) 
                    
                    # Display the exact Ripe Percentage as requested
                    m3.metric("Ripeness Score", f"{ripe_pct:.1f}%") 
                    
                    st.divider()
                    
                    # Detailed List
                    st.write(f"**🌤 Weather Season:** {predicted_season}")
                    st.write(f"**🗓 Current Month:** {current_month_name}") 
                    st.write(f"**📍 Location:** {district}, {state}")
                    st.write(f"**⏳ Crop Age:** {crop_age} months")
                    st.write(f"**ℹ️ Logic Context:** {phase_reason}")
                    
                    # Final Decision
                    if ml_status == "OVERRIPE":
                        st.error("🔴 OVERRIPE / HARVEST IMMEDIATELY")
                    elif ml_status == "RIPE":
                        st.success("🟢 RIPE / READY SOON")
                    else:
                        st.warning("🟡 UNRIPE / NOT READY")