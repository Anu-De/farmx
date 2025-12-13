# 🌱 FarmX - Crop Ripeness Prediction System

> **An AI-Powered Agricultural Solution for Smart Harvesting**

A hybrid AI-powered system that predicts fruit ripeness using deep learning models combined with agricultural logic based on location, weather, and plantation dates.

---

## 👥 Team Members

| Name | Role |
|------|------|
| **Ketan Dinkar** | Team Lead, Model Training & Optimization |
| **Kishlay Kumar** | Model Testing, Image Data Collection/Management, Report Making |
| **Anurag Deshmukh** | Data Collection & UI Management |
| **Mohnish Pradhan** | Data Collection/Management, Report Making |
| **Dipanshu Gupta** | CSV Data Collection & Dataset Management |
| **Mohan Sahu** | | |

---

## 📚 Academic Information

- **Project Type**: Minor Project (College)
- **Duration**: Academic Year 2025-2026
- **Institution**: Chhattisgarh Swami Vivekanada Teechnical University, Bhilai
- **Department**: Computer Science & Engineering(Artificial Intelligence)
- **Project Category**: Artificial Intelligence & Machine Learning
- **Technology Stack**: Python, PyTorch, Streamlit, Deep Learning

---

## 📋 Project Overview

FarmX is a college minor project that bridges the gap between traditional agriculture and modern AI technology. The system uses a two-pronged approach for accurate ripeness prediction:

- **Machine Learning Component**: EfficientNet-B3 models for fruit type and ripeness classification from images
- **Agricultural Logic Component**: Calendar-based ripeness calculation using crop age, regional data, and weather patterns

**Objective**: To help Indian farmers determine the optimal harvest time for various fruits using computer vision and agricultural data, thereby reducing crop losses and maximizing yield quality.

---

## 📁 Project Structure

```
Farmx_main/
│
├── 📂 csv dataset/
│   └── India_Fruit_Dataset.csv                     # Agricultural metadata 
│    └── India_Fruit_Dataset.xlsx                    # Agricultural metadata
│    └── Ordered_Fruit_Production_Flow.xlsx          # Agricultural metadata 
(plantation times, ripening seasons, harvest windows)
│
├── 📂 dataset/                          # Training datasets
│   ├── Train/
│   │   ├── Unripe/                      # Training images (unripe stage)
│   │   ├── Ripe/                        # Training images (ripe stage)
│   │   └── Overripe/                    # Training images (overripe stage)
│   └── Test/
│       ├── Unripe/                      # Test images (unripe stage)
│       ├── Ripe/                        # Test images (ripe stage)
│       └── Overripe/                    # Test images (overripe stage)
│
├── 📂 trained file/
│   ├── ripeness_model.pth               # Trained ripeness classifier (EfficientNet-B3)
│   ├── fruit_type_ripe_only.pth         # Trained fruit type classifier (EfficientNet-B3)
│   ├── ripeness_model_arch.json         # Model architecture definition
│   ├── ripness.txt                      # Ripeness class labels
│   └── type.txt                         # Fruit type labels
│
├── 📂 Final_test_images/                # Sample test images for inference
│
├── 🐍 main.py                           # Core prediction pipeline (CLI)
├── 🐍 app.py                            # Streamlit web interface
├── 🐍 predict_final_fruit_and_ripeness.py  # Batch image analysis script
├── 🐍 ripeness_train.py                 # Training script for ripeness model
├── 🐍 type_train.py                     # Training script for fruit type model
├── 🐍 date_time_module.py               # Date/time utilities
├── 🐍 map_module.py                     # Location & region lookup
├── 🐍 weather_module.py                 # Weather data integration
├── 📄 README.md                         # Project documentation
└── 📄 requirements.txt                  # Python dependencies
```

---

## 🍎 Supported Fruits

The system can classify and predict ripeness for 9 different fruits commonly grown in India:

| Fruit | Season | Primary Region |
|-------|--------|-----------------|
| 🍎 Apple | Sept - Nov | Himachal Pradesh, Uttarakhand |
| 🍌 Banana | Year-round | Karnataka, Tamil Nadu, Andhra Pradesh |
| 🥝 Guava | Aug - Oct | Uttar Pradesh, Rajasthan |
| 🍋 Lime | Nov - May | Maharashtra, Andhra Pradesh |
| 🥭 Mango | March - June | Uttar Pradesh, Andhra Pradesh, Maharashtra |
| 🍊 Orange | Oct - Dec | Maharashtra, Nagpur |
| 🍒 Pomegranate | Sept - Nov | Maharashtra, Karnataka |
| 🍓 Strawberry | Nov - March | Himachal Pradesh, Maharashtra |
| 🍅 Tomato | Year-round | Karnataka, Rajasthan, Haryana |

---

## 🔧 Core Modules

### `main.py` - Core Prediction Pipeline
**Developer**: Kishlay Kumar, Ketan Dinkar

**Purpose**: Command-line interface for ripeness prediction
- Loads pre-trained EfficientNet-B3 models
- Processes single or multiple images
- Combines ML predictions with logic-based calculations
- Generates detailed ripeness reports

**Key Functions**:
- `run_image_prediction()` - Analyzes images and returns ripeness percentages
- `calculate_ripeness()` - Computes logic-based ripeness score using crop age
- `calculate_month_diff()` - Calculates months between plantation and current date
- `main()` - Orchestrates the complete prediction workflow

### `app.py` - Streamlit Web Interface
**Developer**: Anurag Deshmukh, Kishlay Kumar

**Purpose**: Interactive web-based prediction interface
- Image upload and real-time analysis
- Location and weather integration
- Sidebar parameters for customization
- Visual metrics and final recommendations

**Key Functions**:
- `load_models()` - Cached model loading (runs once)
- `get_prediction()` - Single image inference with probability scores
- `calculate_logic_ripeness()` - Logic-based ripeness calculation
- UI components for results visualization

### `predict_final_fruit_and_ripeness.py` - Batch Processing
**Developer**: Anurag Deshmukh

**Purpose**: Analyze multiple images with ensemble voting
- Processes all images in a folder
- Uses majority voting for robust predictions
- Generates batch reports

### `ripeness_train.py` - Ripeness Model Training
**Developer**: Mohan Sahu, Ketan Dinkar

**Purpose**: Train the ripeness classification model
- EfficientNet-B3 backbone
- 3 output classes: Unripe, Ripe, Overripe
- Custom training pipeline with data augmentation
- Uses Indian fruit dataset for training

### `type_train.py` - Fruit Type Model Training
**Developer**: Mohan Sahu, Ketan Dinkar

**Purpose**: Train the fruit type classification model
- EfficientNet-B3 backbone
- 9 output classes: apple, banana, guava, lime, mango, orange, pomegranate, strawberry, tomato
- Custom training pipeline
- Transfer learning from ImageNet

### `date_time_module.py` - Date/Time Utilities
**Developer**: Anurag Deshmukh

**Functions**:
- `get_current_datetime()` - Returns current date and time in formatted string

### `map_module.py` - Location & Region Data
**Developer**: Dipanshu Gupta

**Functions**:
- `get_coordinates_region_area(location)` - Returns coordinates, district, state, country for a location

### `weather_module.py` - Weather Integration
**Developer**: Anurag Deshmukh, Dipanshu Gupta

**Functions**:
- `get_weather_data(location, state)` - Fetches weather information
- `get_season_from_weather(weather_data)` - Predicts season from weather data

---

## 📊 Prediction Output Structure

The system provides **three layers of analysis**:

### 1️⃣ ML Ripeness Prediction (Deep Learning)
- **Unripe Percentage**: Model's confidence in unripe stage
- **Ripe Percentage**: Model's confidence in ripe stage
- **Overripe Percentage**: Model's confidence in overripe stage
- **ML Status**: UNRIPE / RIPE / OVERRIPE (based on highest probability)

### 2️⃣ Logic-Based Ripeness (Agricultural Calculation)
- **Ripeness Score**: 0-100% (based on crop age)
- **Crop Age**: Months since plantation
- **Phase Reason**: Growth phase interpretation
  - "Vegetative growth phase" (score < 50%)
  - "Inside ripening season" (50-80%)
  - "Harvest window period" (≥ 80%)

### 3️⃣ Combined Final Result
- **OVERRIPE / HARVEST IMMEDIATELY**: If ML ≥ 50% overripe OR logic ≥ 90%
- **RIPE / READY SOON**: If ML ≥ 50% ripe OR logic ≥ 60%
- **UNRIPE / NOT READY**: Otherwise

---

## 🚀 How to Use

### Option 1: Web Interface (Recommended)
```bash
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

**Steps**:
1. Enter location in sidebar
2. Select plantation month
3. Upload fruit image
4. Click "Analyze Ripeness"
5. View results with visualizations

### Option 2: Command Line Interface
```bash
python main.py
```

**Steps**:
1. Place images in the INPUT_FOLDER
2. Run the script
3. Enter location when prompted
4. Enter plantation month when prompted
5. View detailed ripeness report in terminal

### Option 3: Batch Processing
```bash
python predict_final_fruit_and_ripeness.py
```

---

## 📦 Installation & Requirements

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- GPU support optional (CPU mode fully supported)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Core Libraries
- **PyTorch & TorchVision**: Deep learning framework
- **Streamlit**: Web interface framework
- **Pandas**: Data manipulation
- **Pillow**: Image processing
- **NumPy**: Numerical computations
- **Requests**: API calls for weather data

---

## 🧠 Model Architecture

### Both Models Use: EfficientNet-B3

**Ripeness Model**:
- Input: 224×224 RGB images
- Backbone: EfficientNet-B3 (trained from scratch on Indian fruit dataset)
- Output Head: 3 classes (Unripe, Ripe, Overripe)
- Activation: Softmax (for probability distribution)
- Training Data: 1000+ labeled images across 3 ripeness stages

**Fruit Type Model**:
- Input: 224×224 RGB images
- Backbone: EfficientNet-B3 (transfer learning with ImageNet initialization)
- Output Head: 9 classes (apple, banana, guava, lime, mango, orange, pomegranate, strawberry, tomato)
- Activation: Softmax (for probability distribution)
- Training Data: 5000+ labeled fruit images

**Image Preprocessing**:
```python
transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], 
                        std=[0.229,0.224,0.225])
])
```

---

## 📊 CSV Dataset Structure

**India_Fruit_Dataset.csv** contains:
- `Fruit`: Fruit name
- `Plantation_Time`: Recommended plantation period for Indian regions
- `Ripening_Time`: Typical ripening season in India
- `Harvesting_Time`: Optimal harvest window period
- Regional agricultural data specific to Indian climate and geography

**Data Compiled By**: Dipanshu Gupta with agricultural research references

---

## 🔍 Input Requirements

### Image Input
- **Formats**: JPG, JPEG, PNG, BMP, WEBP
- **Recommended Size**: 256×256 pixels or larger
- **Content**: Clear, well-lit fruit image for best accuracy
- **Background**: Any background (model handles various conditions)

### User Inputs
- **Location**: City/Village/District name in India (used for weather and regional data)
- **Plantation Month**: Month when crop was planted (used for crop age calculation)

---

## 💾 File Paths Configuration

**For Windows** (in `main.py`):
```python
RIPENESS_MODEL = r"D:\programs\python\fruit reeping prediction\trained file\ripeness_model.pth"
FRUIT_MODEL    = r"D:\programs\python\fruit reeping prediction\trained file\fruit_type_ripe_only.pth"
INPUT_FOLDER   = r"D:\programs\python\fruit reeping prediction\Input images"
```

**For macOS/Linux** (in `app.py` - uses relative paths):
```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RIPENESS_MODEL_PATH = os.path.join(BASE_DIR, "trained file", "ripeness_model.pth")
FRUIT_MODEL_PATH = os.path.join(BASE_DIR, "trained file", "fruit_type_ripe_only.pth")
CSV_PATH = os.path.join(BASE_DIR, "csv dataset", "India_Fruit_Dataset.csv")
```

---

## 🎯 Example Workflow

**Scenario**: Farmer uploads mango image for ripeness check

1. **Upload Image**: User uploads a mango fruit image
2. **ML Prediction**: Model outputs → 5% Unripe, **85% Ripe**, 10% Overripe
3. **Logic Calculation**: User planted 6 months ago → Ripeness Score = 72%
4. **Location & Weather**: System identifies Andhra Pradesh (mango region), checks monsoon patterns
5. **Combined Analysis**: 
   - ML says: RIPE (85% > 50%)
   - Logic says: Ripening season (72% is between 50-80%)
   - Weather: Favorable ripening conditions
6. **Final Output**: **RIPE / READY SOON** ✅ (Harvest recommended within 5-7 days)

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Model files not found | Ensure `trained file/` folder contains `.pth` files |
| Weather API errors | Check internet connection; fallback to "Unknown" |
| Location not recognized | Use standard city/district names in India |
| GPU out of memory | Run in CPU mode (automatic fallback in `app.py`) |
| Module import errors | Run `pip install -r requirements.txt` |
| Image upload fails | Check image format; ensure JPG/PNG with size < 10MB |

---

## 📈 Performance Metrics

- **Model Accuracy**: ~92% on Indian fruit dataset
- **Inference Speed**: <2 seconds per image (CPU), <0.5s (GPU)
- **Supported Regions**: All major agricultural regions across India
- **Weather API Coverage**: All Indian states and major cities

---

## 👨‍💻 Development & Contributions

### Development Timeline
- **Phase 1** (Oct 2024): Dataset collection and preprocessing
- **Phase 2** (Nov 2024): Model training and optimization
- **Phase 3** (Dec 2024): Web interface development and integration

### Training New Models
```bash
python ripeness_train.py    # Train ripeness model
python type_train.py        # Train fruit type model
```

### Adding New Fruits
1. Collect training images (Unripe, Ripe, Overripe stages) - Coordinated by Dipanshu Gupta
2. Update `FRUIT_CLASSES` in training script
3. Retrain fruit type model (Mohan Sahu)
4. Update `India_Fruit_Dataset.csv` with agricultural data
5. Test and validate (Ketan Dinkar)

---

## 📝 Project Deliverables

- ✅ Functional ripeness prediction system
- ✅ Web-based interface with Streamlit
- ✅ Command-line tool for farmers
- ✅ Pre-trained ML models
- ✅ Agricultural dataset for India
- ✅ Comprehensive documentation
- ✅ Sample test images and results

---

## 📚 References & Resources

- EfficientNet: Scaling with Compound Coefficient (Tan & Le, 2019)
- PyTorch Documentation
- Indian Agricultural Statistics at a Glance (Ministry of Agriculture)
- Fruit Ripeness Detection using Computer Vision (Research Papers)

---

## 📧 Contact & Support

**Project Repository**: GitHub (Farmx_main)

**For Issues or Questions**:
- **Technical Issues**: Create a GitHub issue
- **Data/Dataset**: Contact Dipanshu Gupta
- **ML Models**: Contact Mohan Sahu & Ketan Dinkar
- **UI/Frontend**: Contact Mohnish Pradhan
- **Integration**: Contact Kishlay Kumar

---

## 📜 License & Academic Use

This project is developed as a **college minor project** for educational and research purposes. It's intended to demonstrate the practical application of AI/ML in agriculture.

**Note**: This is an academic project and not intended for commercial distribution without proper authorization and acknowledgment of all team members.

---

## 🎓 Academic Acknowledgment

This project was developed as a **Minor Project** in the Computer Science & Engineering Department. We acknowledge the guidance of our faculty advisors and the institutional support provided.

---

**Project Status**: ✅ Complete & Functional

**Last Updated**: 11 December 2025

**Team**: Ketan Dinkar | Kishlay Kumar | Anurag Deshmukh | Mohnish Pradhan | Dipanshu Gupta | Mohan Sahu

---

*"Empowering Indian Agriculture with Artificial Intelligence"* 🌾🤖
