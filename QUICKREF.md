# 🚀 Quick Reference Guide

> **Fast access to key information about the Predictive Modeling projects**

---

## 📂 File Locations

### Diabetes Project
```bash
Diabetes/
├── Pima.ipynb              # Main notebook
├── diabetes.csv            # Dataset (768 records)
├── README.md               # Full documentation
└── requirements.txt        # Dependencies
```

### CKD Project
```bash
Kidney Disease/
├── CKD_Analysis.ipynb      # Main notebook
├── kidney_disease.csv      # Dataset (400 records)
├── README.md               # Full documentation
├── requirements.txt        # Dependencies
├── model_*.pkl             # Saved models
├── scaler.pkl              # Feature scaler
└── *.json                  # Results & reports
```

---

## ⚡ Quick Start Commands

### Setup
```bash
# Clone repository
git clone https://github.com/worldisconfusion/Predictive-Modeling-for-Diabetes-and-Chronic-Kidney-Disease-Detection.git
cd Predictive-Modeling-for-Diabetes-and-Chronic-Kidney-Disease-Detection

# For Diabetes project
cd Diabetes
pip install -r requirements.txt
jupyter notebook Pima.ipynb

# For CKD project
cd "Kidney Disease"
pip install -r requirements.txt
jupyter notebook CKD_Analysis.ipynb
```

### Load Pre-trained Models (CKD only)
```python
import pickle

# Load best model
with open('Kidney Disease/best_model_logistic_regression.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open('Kidney Disease/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
```

---

## 📊 Dataset Quick Facts

| Aspect | Diabetes (PIMA) | CKD |
|--------|----------------|-----|
| **Records** | 768 | 400 |
| **Features** | 8 original | 24 original |
| **Engineered** | 6 features | 7 features |
| **Missing Data** | Up to 48.7% | Up to 40% |
| **Target** | Binary (0/1) | Binary (0/1) |
| **Class Balance** | 65% / 35% | 62.5% / 37.5% |
| **Population** | PIMA Indian females 21+ | Mixed population |

---

## 🎯 Model Performance Cheat Sheet

### Diabetes Models

| Model | Accuracy | ROC-AUC | When to Use |
|-------|----------|---------|-------------|
| **XGBoost** ⭐ | 83.0% | 0.901 | Best overall performance |
| **Random Forest** | 81.5% | 0.888 | Good balance, interpretable |
| **Logistic Reg.** | 76.5% | 0.843 | Fast, explainable baseline |

### CKD Models

| Model | Accuracy | ROC-AUC | When to Use |
|-------|----------|---------|-------------|
| **Logistic Reg.** ⭐ | 100% | 1.000 | Best, saved model available |
| **k-NN** | 98.8% | 0.999 | Instance-based learning |
| **Naive Bayes** | 95.0% | 1.000 | Fast probabilistic model |

---

## 🔑 Key Features by Importance

### Diabetes (Top 5)
1. **Glucose** (30%) - Plasma glucose concentration
2. **BMI** (18%) - Body Mass Index
3. **Age** (15%) - Patient age
4. **DiabetesPedigreeFunction** (11%) - Genetic influence
5. **Insulin_Log** (9%) - Log-transformed insulin

### CKD (Key Biomarkers)
- **Hemoglobin** - Anemia indicator
- **Serum Creatinine** - Kidney function
- **Blood Urea** - Waste filtration
- **Albumin** - Protein levels
- **Specific Gravity** - Urine concentration

---

## 🧪 Statistical Tests Applied

### Diabetes
- ✅ Mann-Whitney U (Glucose vs Diabetes)
- ✅ Spearman Correlation (BMI vs Diabetes)
- ✅ Independent t-test (Age vs Diabetes)

### CKD
- ✅ t-test & Mann-Whitney U (Hemoglobin)
- ✅ Chi-square (Hypertension association)
- ✅ ANOVA & Kruskal-Wallis (Blood urea)
- ✅ Spearman (Specific gravity correlation)
- ✅ Pearson (Hemoglobin-Creatinine)

---

## 📈 Preprocessing Pipeline Summary

### Diabetes
```
1. Load data (768 × 9)
2. Handle missing values (median imputation)
3. Detect & treat outliers (IQR + Winsorization)
4. Engineer features (+6 features → 14 total)
5. Scale features (StandardScaler)
6. Balance classes (SMOTE: 500 vs 500)
7. Split data (80/20 stratified)
```

### CKD
```
1. Load data (400 × 25)
2. Handle missing values (median/mode)
3. Encode categoricals (binary + ordinal)
4. Detect & treat outliers (IQR capping)
5. Engineer features (+7 features → 29 total)
6. Scale features (StandardScaler)
7. Split data (80/20 stratified)
8. Save preprocessed data & artifacts
```

---

## 🛠️ Common Code Snippets

### Load Dataset
```python
import pandas as pd

# Diabetes
df_diabetes = pd.read_csv('Diabetes/diabetes.csv')

# CKD
df_ckd = pd.read_csv('Kidney Disease/kidney_disease.csv')
```

### Make Predictions (CKD)
```python
import pickle
import pandas as pd

# Load artifacts
with open('Kidney Disease/best_model_logistic_regression.pkl', 'rb') as f:
    model = pickle.load(f)
with open('Kidney Disease/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# New patient data (example)
new_patient = pd.DataFrame({
    'age': [55], 'bp': [80], 'bgr': [121],
    'bu': [42], 'sc': [1.3], 'hemo': [12.6],
    # ... all 29 features required
})

# Predict
scaled_data = scaler.transform(new_patient)
prediction = model.predict(scaled_data)[0]
probability = model.predict_proba(scaled_data)[0][1]

print(f"Prediction: {'CKD' if prediction == 1 else 'No CKD'}")
print(f"Probability: {probability:.2%}")
```

### Feature Importance
```python
import matplotlib.pyplot as plt

# For tree-based models (Random Forest, XGBoost)
feature_importance = model.feature_importances_
features = df.columns[:-1]  # Exclude target

# Plot
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importance)
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()
```

---

## 📚 Documentation Navigation

| Document | Purpose | Lines |
|----------|---------|-------|
| **README.md** | Main overview, quick start | 718 |
| **ARCHITECTURE.md** | Technical deep-dive | 1,007 |
| **Diabetes/README.md** | Diabetes project details | 917 |
| **Kidney Disease/README.md** | CKD project details | 178 |
| **QUICKREF.md** | This document | ~200 |

---

## 🔍 Finding Things

### Want to understand the project?
→ Start with **README.md**

### Want technical details?
→ Read **ARCHITECTURE.md**

### Want to use Diabetes models?
→ Open **Diabetes/Pima.ipynb**

### Want to use CKD models?
→ Open **Kidney Disease/CKD_Analysis.ipynb**

### Want to see preprocessing details?
→ Check **Kidney Disease/preprocessing_report.json**

### Want to see model results?
→ Check **Kidney Disease/model_results.json**

---

## ⚠️ Important Warnings

### ❌ NOT for Clinical Use
These models are:
- Educational prototypes
- Not FDA/CE approved
- Require professional supervision
- Need external validation

### ⚠️ Population Limitations
- **Diabetes**: PIMA Indian females only
- **CKD**: Mixed population
- May not generalize to all demographics

### 🔒 Privacy Considerations
- Datasets are anonymized
- HIPAA compliance required for deployment
- No PHI in training data

---

## 🐛 Troubleshooting

### Issue: Missing Dependencies
```bash
# Solution
pip install -r requirements.txt
```

### Issue: Can't Load .pkl Files
```python
# Solution: Ensure same Python version
import sys
print(f"Python version: {sys.version}")
# Should be Python 3.8+
```

### Issue: Out of Memory
```python
# Solution: Close other notebooks
# Or reduce dataset size
df_sample = df.sample(frac=0.5, random_state=42)
```

### Issue: Notebook Won't Run
```bash
# Solution: Ensure Jupyter is installed
pip install jupyter notebook
jupyter notebook
```

---

## 📞 Getting Help

### For Code Issues
- Check **ARCHITECTURE.md** for technical details
- Review notebook cells for inline comments
- Check **requirements.txt** for dependencies

### For Methodology Questions
- Read project-specific **README.md** files
- Review statistical testing sections
- Check references in documentation

### For Contribution
- See **Contributing** section in main README.md
- Follow code standards (PEP 8)
- Add tests for new features

### For General Questions
- GitHub Issues: [Create an issue](https://github.com/worldisconfusion/Predictive-Modeling-for-Diabetes-and-Chronic-Kidney-Disease-Detection/issues)
- Repository: [View on GitHub](https://github.com/worldisconfusion/Predictive-Modeling-for-Diabetes-and-Chronic-Kidney-Disease-Detection)

---

## 🎓 Learning Path

1. **Beginner**: Start with README.md → Explore one notebook
2. **Intermediate**: Complete one full notebook → Read ARCHITECTURE.md
3. **Advanced**: Run both notebooks → Modify preprocessing → Tune models
4. **Expert**: Deploy models → Add new features → Contribute back

---

## 📦 Dependencies (Core)

```bash
# Data & ML
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Statistics
scipy>=1.7.0
statsmodels>=0.13.0

# Explainability
shap>=0.40.0

# Development
jupyter>=1.0.0
```

---

## ✅ Checklist for New Users

- [ ] Clone repository
- [ ] Install dependencies
- [ ] Read main README.md
- [ ] Choose a project (Diabetes or CKD)
- [ ] Open corresponding notebook
- [ ] Run cells sequentially
- [ ] Review visualizations
- [ ] Check model performance
- [ ] Read technical documentation (optional)
- [ ] Experiment with hyperparameters (optional)

---

## 🎯 Common Tasks

### Task: Run Complete Analysis
```bash
cd Diabetes
jupyter notebook Pima.ipynb
# Run all cells: Cell → Run All
```

### Task: Export Results
```python
# Save predictions
predictions.to_csv('predictions.csv', index=False)

# Save plots
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
```

### Task: Compare Models
```python
# Use model comparison table in notebooks
results_df = pd.DataFrame({
    'Model': [...],
    'Accuracy': [...],
    'ROC-AUC': [...]
})
print(results_df.sort_values('ROC-AUC', ascending=False))
```

---

**Last Updated**: November 2024  
**Version**: 1.0  
**Quick Ref Doc for**: Predictive Modeling Project

---

<p align="center">
  <i>For detailed information, see README.md and ARCHITECTURE.md</i>
</p>
