# ❤️ CardioScan: Heart Disease Prediction System

A comprehensive web-based application for predicting heart disease risk using a **Random Forest classifier built from scratch**. Features an interactive dashboard for model evaluation and hyperparameter tuning.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Setup Instructions](#setup-instructions)
- [Running the Application](#running-the-application)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

**CardioScan** predicts the likelihood of heart disease in patients based on clinical parameters. The application combines:

- **Custom Random Forest Implementation**: A from-scratch implementation of the Random Forest algorithm (no scikit-learn RF)
- **Web Interface**: Flask-based UI for ease of use
- **Interactive Evaluation Dashboard**: Real-time hyperparameter tuning with visual feedback
- **Machine Learning Pipeline**: Complete data preprocessing, training, and evaluation workflow

### Dataset
- **Source**: UCI Heart Disease Dataset
- **Samples**: 303 patient records
- **Features**: 13 clinical parameters (age, sex, cholesterol, blood pressure, etc.)
- **Target**: Binary classification (0 = no disease, 1 = disease present)

---

## ✨ Key Features

### 🏥 Patient Prediction
- User-friendly web interface for entering patient vitals
- 13 clinical input fields (age, sex, chest pain type, blood pressure, etc.)
- Real-time prediction with confidence assessment
- Instant results displayed on the UI

### 📊 Interactive Model Evaluation Dashboard
- **Live Hyperparameter Tuning**:
  - Number of Trees: 10-500 (slider)
  - Maximum Tree Depth: 2-20 (slider)
  - Minimum Samples Split: 2-50 (slider)

- **Real-time Metrics Display**:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Sensitivity
  - Specificity
  - ROC-AUC Score

- **Visualization Tools**:
  - Seaborn-style confusion matrix heatmap
  - ROC curve for model assessment
  - Parameter comparison charts
  - Performance trend analysis

### 🔬 Advanced Evaluation Features
- Train temporary models with custom parameters without affecting production model
- Fair comparison methodology using consistent test sets
- Parameter sensitivity analysis with comparison charts
- Export evaluation results

---

## 💻 System Requirements

### Minimum Requirements
- **OS**: macOS, Linux, or Windows (with WSL)
- **Python**: 3.8 or higher
- **RAM**: 4GB (8GB recommended)
- **Disk Space**: 2GB (includes dependencies and datasets)

### Browser Support
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

---

## 🚀 Setup Instructions

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd "ai_project 2"
```

### Step 2: Create Virtual Environment
Create an isolated Python environment to avoid dependency conflicts:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

Your terminal prompt should now show `(.venv)` indicating the virtual environment is active.

### Step 3: Install Dependencies
```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

**Required Packages**:
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning utilities (metrics, preprocessing)
- `matplotlib` - Visualization library
- `seaborn` - Statistical data visualization
- `flask` - Web framework for the application

### Step 4: Verify Installation
```bash
# Test if all dependencies are installed correctly
python -c "import flask, pandas, numpy, matplotlib, seaborn, sklearn; print('✅ All dependencies installed successfully!')"
```

### Step 5: Prepare Data
The dataset files are included in the `data/` directory:
- `heart.csv` - Main dataset file
- `heart2.csv` - Alternative dataset
- `heart_kaggle.csv` - Kaggle version

No additional data preparation is required. The application automatically handles:
- Data loading
- Feature normalization
- Train/test split (70/30)

---

## 🎮 Running the Application

### Start the Flask Server

```bash
# Ensure virtual environment is activated (you should see (.venv) in prompt)
source .venv/bin/activate

# Navigate to project directory
cd "ai_project 2"

# Start the Flask development server
python app/app.py
```

You should see output like:
```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:8080
 * Press CTRL+C to quit
```

### Access the Application

Open your web browser and navigate to:
```
http://localhost:8080
```

The application will load with two main tabs:
1. **🏥 Patient Prediction** - Make predictions for individual patients
2. **📊 Model Evaluation** - Evaluate and tune the model

### Stop the Server

Press `CTRL+C` in the terminal running the Flask server.

---

## 📖 Usage Guide

### Tab 1: Patient Prediction

**Input Fields** (enter patient data):
1. **Age** (years): 29-77
2. **Sex**: 0 = Female, 1 = Male
3. **Chest Pain Type** (cp): 0-3
   - 0 = Typical Angina
   - 1 = Atypical Angina
   - 2 = Non-anginal Pain
   - 3 = Asymptomatic
4. **Resting Blood Pressure** (trestbps): 90-200 mmHg
5. **Cholesterol** (chol): 126-564 mg/dL
6. **Fasting Blood Sugar** (fbs): 0 = <120 mg/dL, 1 = ≥120 mg/dL
7. **Resting ECG** (restecg): 0-2
8. **Max Heart Rate** (thalach): 60-202 bpm
9. **Exercise Induced Angina** (exang): 0 = No, 1 = Yes
10. **ST Depression** (oldpeak): 0-6.2
11. **ST Slope** (slope): 0-2
12. **Number of Vessels** (ca): 0-4
13. **Thalassemia** (thal): 0-3

**Steps**:
1. Fill in all 13 fields with patient data
2. Click **"Predict"** button
3. View prediction result:
   - **Risk Score**: 0-100% probability of heart disease
   - **Prediction**: "Heart Disease Detected" or "No Heart Disease"
   - **Confidence**: High/Medium/Low

### Tab 2: Model Evaluation Dashboard

**Features**:

1. **Parameter Sliders** (adjust model hyperparameters):
   - **Number of Trees**: Controls ensemble size (10-500)
     - Lower = Faster, less accurate
     - Higher = Slower, more accurate
   - **Max Tree Depth**: Controls tree complexity (2-20)
     - Lower = Simpler, less overfitting
     - Higher = More complex, risk of overfitting
   - **Min Samples Split**: Minimum samples to split node (2-50)
     - Higher = Simpler tree, less overfitting

2. **Buttons**:
   - **"Evaluate Model"**: Train temporary model with current parameters and view metrics
   - **"Compare Parameters"**: Compare how each parameter affects model accuracy

3. **Metrics Cards** (displayed after evaluation):
   - **Accuracy**: Overall correctness (TP+TN)/(Total)
   - **Precision**: Of predicted positive, how many are correct
   - **Recall**: Of actual positive, how many are detected
   - **F1-Score**: Harmonic mean of precision and recall
   - **Sensitivity**: True positive rate
   - **Specificity**: True negative rate
   - **ROC-AUC**: Area under ROC curve (0.5-1.0)

4. **Visualizations**:
   - **Confusion Matrix**: Shows TP, TN, FP, FN distribution
   - **ROC Curve**: Trade-off between TPR and FPR
   - **Comparison Charts**: Parameter impact analysis

**Workflow Example**:
```
1. Adjust "Number of Trees" slider to 200
2. Adjust "Max Depth" slider to 10
3. Click "Evaluate Model"
4. View accuracy: 89.76%
5. Click "Compare Parameters"
6. See how changing trees affects accuracy
7. Fine-tune parameters for best performance
```

---

## 📁 Project Structure

```
ai_project 2/
├── readme.md                 # This file - complete documentation
├── requirements.txt          # Python dependencies
├── main.py                   # Legacy/alternative entry point
│
├── app/
│   ├── app.py               # Flask application & routes
│   └── templates/
│       └── index.html       # Web UI (prediction & evaluation)
│
├── data/
│   ├── heart.csv            # Main dataset (UCI Heart Disease)
│   ├── heart2.csv           # Alternative dataset
│   └── heart_kaggle.csv     # Kaggle version
│
├── src/
│   ├── train.py             # Model training script
│   ├── predict.py           # Prediction functions
│   ├── evaluate.py          # Evaluation metrics
│   ├── evaluate_params.py   # Hyperparameter evaluation engine
│   ├── utils/
│   │   ├── data_preprocessing.py  # Data cleaning & normalization
│   │   └── metrics.py            # Custom metric calculations
│   └── models/
│       ├── random_forest.py      # Random Forest implementation
│       └── decision_tree.py      # Decision Tree (base learner)
│
├── models/                  # Saved model checkpoints
├── notebooks/               # Jupyter notebooks (if any)
├── reports/                 # Generated reports & analysis
└── .venv/                   # Virtual environment (auto-created)
```

---

## 🔬 How It Works

### Model Architecture

**Random Forest**:
1. **Ensemble Learning**: Combines multiple decision trees
2. **Bootstrap Aggregating**: Each tree trained on random subset of data
3. **Feature Randomness**: Random feature selection at each split
4. **Majority Voting**: Final prediction = most common tree prediction

### Data Pipeline

```
Raw Data (heart.csv)
    ↓
[Preprocessing]
  - Handle missing values
  - Feature normalization (0-1 scale)
  - Duplicate removal
    ↓
[Train/Test Split]
  - 70% training data (212 samples)
  - 30% testing data (91 samples)
    ↓
[Model Training]
  - Fit Random Forest (100 trees by default)
  - Optimize splits with information gain
    ↓
[Evaluation]
  - Test on held-out data
  - Calculate metrics (accuracy, precision, etc.)
    ↓
[Predictions]
  - Accept new patient data
  - Normalize inputs
  - Get forest predictions
  - Return risk score
```

### Feature Importance

The model uses these 13 clinical features:
- **Demographic**: Age, Sex
- **Symptom**: Chest Pain Type
- **Vital Signs**: Resting BP, Max Heart Rate
- **Clinical**: Cholesterol, Fasting Blood Sugar
- **Test Results**: ECG, ST Depression, ST Slope
- **Diagnostic**: Number of Vessels, Thalassemia

---

## 📜 File Descriptions

### Core Application Files

| File | Purpose |
|------|---------|
| `app/app.py` | Flask server with routes for prediction & evaluation |
| `app/templates/index.html` | Web UI: patient form & evaluation dashboard |
| `requirements.txt` | Python package dependencies |

### Machine Learning Pipeline

| File | Purpose |
|------|---------|
| `src/train.py` | Load data, train Random Forest, save model |
| `src/predict.py` | Load model, make predictions for new patients |
| `src/evaluate.py` | Calculate metrics (accuracy, precision, recall, etc.) |
| `src/evaluate_params.py` | Temporary model training for hyperparameter tuning |

### Model Implementation

| File | Purpose |
|------|---------|
| `src/models/random_forest.py` | Random Forest classifier from scratch |
| `src/models/decision_tree.py` | Decision Tree (base learner for RF) |

### Utilities

| File | Purpose |
|------|---------|
| `src/utils/data_preprocessing.py` | Data loading, normalization, splitting |
| `src/utils/metrics.py` | Confusion matrix, ROC curve, evaluation metrics |

### Data Files

| File | Purpose |
|------|---------|
| `data/heart.csv` | UCI Heart Disease dataset (303 samples) |
| `data/heart2.csv` | Alternative heart disease dataset |
| `data/heart_kaggle.csv` | Kaggle version of dataset |

---

## 🔧 Troubleshooting

### Issue: "Command not found: python"
**Solution**: 
```bash
# Activate virtual environment first
source .venv/bin/activate

# Then run Flask
python app/app.py
```

### Issue: "ModuleNotFoundError: No module named 'flask'"
**Solution**:
```bash
# Ensure virtual environment is activated (see (.venv) in prompt)
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "Address already in use Port 8080"
**Solution**:
```bash
# Kill process using port 8080
# On macOS/Linux:
lsof -i :8080 | grep LISTEN | awk '{print $2}' | xargs kill -9

# Then start Flask again
python app/app.py
```

### Issue: "Browser can't connect to localhost:8080"
**Solution**:
1. Check Flask is running (terminal shows `Running on http://127.0.0.1:8080`)
2. Verify port 8080 is not blocked by firewall
3. Try accessing `http://127.0.0.1:8080` instead of `localhost`
4. Open browser console (F12) and check for errors

### Issue: "Evaluation failed" in Model Evaluation dashboard
**Solution**:
1. Check console for errors: Open browser DevTools (F12)
2. Check Flask terminal for error messages
3. Ensure data files exist: `ls data/heart.csv`
4. Restart Flask application

### Issue: Slow predictions or evaluations
**Solution**:
1. Model evaluation trains temporary models - this is normal and takes ~2-5 seconds
2. For faster evaluation, reduce "Number of Trees" to 50-100
3. Patient predictions should be instant
4. Ensure no other heavy processes are running

### Issue: Conda/Virtual environment issues
**Solution**:
```bash
# Create fresh virtual environment
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 Performance Metrics

### Model Baseline (Default: 100 trees, depth=10, min_split=2)
- **Accuracy**: ~89.76%
- **Precision**: ~89.83%
- **Recall**: ~89.76%
- **F1-Score**: ~89.80%
- **ROC-AUC**: ~0.96

### Typical Hyperparameter Tuning Results
| Trees | Depth | Min Split | Accuracy |
|-------|-------|-----------|----------|
| 50 | 5 | 2 | ~87% |
| 100 | 10 | 2 | ~90% |
| 200 | 15 | 5 | ~91% |
| 500 | 20 | 10 | ~92% |

---

## 🎓 Learning Resources

### Understanding Random Forests
- Algorithm randomly selects features at each split
- Reduces variance through ensemble averaging
- Less prone to overfitting than single trees
- Works well with mixed feature types

### Decision Trees (Base Learner)
- Recursive algorithm creating hierarchical splits
- Uses information gain to select best splits
- Each node is a "if-then" rule
- Leaf nodes contain final predictions

### Model Evaluation Metrics
- **Accuracy**: (TP+TN)/(Total) - overall correctness
- **Precision**: TP/(TP+FP) - among predicted positive, how many correct
- **Recall**: TP/(TP+FN) - among actual positive, how many detected
- **F1-Score**: 2*(Precision*Recall)/(Precision+Recall) - harmonic mean
- **ROC-AUC**: Area under ROC curve - 0.5 (random) to 1.0 (perfect)

---

## ⚠️ Medical Disclaimer

**This application is for educational and research purposes only.** It should NOT be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical advice and diagnosis.

---

## 📝 License & Credits

- **Dataset Source**: UCI Heart Disease Dataset
- **Built with**: Python, Flask, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Author**: Ayush
- **Date**: March 2026

---

## 🤝 Support

For issues, questions, or suggestions:
1. Check the [Troubleshooting](#troubleshooting) section above
2. Review error messages in browser console (F12)
3. Check Flask terminal for detailed error logs

---

**Happy predicting! ❤️**

