import random
from datetime import datetime
from date_time_module import get_current_datetime  # Import for month

# Presets for Nagpur (expand for other cities; from historical data)
NAGPUR_PRESETS = {
    11: {"temp": 25, "humidity": 60, "rain": 2},  # November: Warm, dry post-monsoon
    1: {"temp": 22, "humidity": 50, "rain": 1},   # Winter: Mild, dry
    5: {"temp": 38, "humidity": 30, "rain": 0},   # Summer: Hot, dry
    7: {"temp": 28, "humidity": 85, "rain": 100}, # Monsoon: Warm, wet
    # Defaults
    "default": {"temp": 28, "humidity": 60, "rain": 10}
}

def get_weather_data(city, state=None):
    """
    Realistic mock: Preset by city/month, with light randomness.
    """
    if city.lower() == "nagpur":
        presets = NAGPUR_PRESETS
    else:
        presets = {"default": NAGPUR_PRESETS["default"]}  # Expand later
    
    # Get current month from date_time_module
    date_data = get_current_datetime()
    now = datetime.strptime(date_data['date'], "%d-%m-%Y")
    month = now.month
    
    preset = presets.get(month, presets["default"])
    
    weather = {
        "temperature": round(random.uniform(preset["temp"] - 2, preset["temp"] + 2), 1),
        "humidity": round(random.uniform(preset["humidity"] - 10, preset["humidity"] + 10)),
        "rainfall": round(random.uniform(max(0, preset["rain"] - 1), preset["rain"] + 2), 1)  # Low variance for realism
    }
    
    # Optional: Real API for 2025 forecasts (uncomment if online)
    # try:
    #     import requests
    #     url = "https://api.open-meteo.com/v1/forecast?latitude=21.15&longitude=79.08&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&current=relative_humidity_2m&timezone=Asia/Kolkata"
    #     resp = requests.get(url)
    #     data = resp.json()
    #     weather = {
    #         "temperature": round((data['daily']['temperature_2m_max'][0] + data['daily']['temperature_2m_min'][0]) / 2, 1),
    #         "humidity": data['current']['relative_humidity_2m'],
    #         "rainfall": data['daily']['precipitation_sum'][0]
    #     }
    # except:
    #     pass  # Use mock
    
    return weather

def get_season_from_weather(weather):
    """
    Tuned thresholds for daily data (realistic ranges).
    """
    temp = weather["temperature"]
    humidity = weather["humidity"]
    rain = weather["rainfall"]  # Daily: 0-50mm typical

    if rain > 50 and humidity > 75:  # Heavy daily rain
        return "Monsoon"
    elif rain > 5 and 55 <= humidity <= 75:  # Transitional
        return "Post-Monsoon / Early Winter"
    elif temp < 22 and rain < 5:  # Cool + dry
        return "Winter"
    elif temp > 32 and rain < 3:  # Hot + dry
        return "Summer"
    else:  # Mild, low rain (e.g., Nov Nagpur)
        return "Post-Monsoon / Early Winter"  # Default to common transitional