# 🏥 Predictive Modeling for Diabetes and Chronic Kidney Disease Detection

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Educational-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-success)](README.md)

> **A comprehensive machine learning project for predicting diabetes and chronic kidney disease using diagnostic measurements and clinical data.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Projects](#projects)
  - [Diabetes Prediction](#1-diabetes-prediction-pima-indians-dataset)
  - [Chronic Kidney Disease Detection](#2-chronic-kidney-disease-detection)
- [Quick Start](#quick-start)
- [Dataset Information](#dataset-information)
- [Model Performance](#model-performance)
- [Documentation](#documentation)
- [Ethical Considerations](#ethical-considerations)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This repository contains two comprehensive machine learning projects focused on medical diagnosis prediction:

1. **Diabetes Prediction** - Using the PIMA Indians Diabetes Database
2. **Chronic Kidney Disease (CKD) Detection** - Using clinical and diagnostic parameters

Both projects demonstrate end-to-end data science workflows including:
- Exploratory Data Analysis (EDA)
- Statistical Hypothesis Testing
- Data Preprocessing & Feature Engineering
- Multiple ML Model Implementation
- Model Evaluation & Comparison
- Model Explainability (SHAP values)
- Comprehensive Visualization

### 🎓 Purpose

These projects are designed for:
- **Healthcare professionals** seeking diagnostic support tools
- **Data scientists** learning medical ML applications
- **Researchers** studying disease risk factors
- **Students** understanding complete ML pipelines
- **Public health officials** designing screening programs

---

## 📁 Project Structure

```
Predictive-Modeling-for-Diabetes-and-Chronic-Kidney-Disease-Detection/
│
├── README.md                          # This file - Main project documentation
├── .gitignore                         # Git ignore configuration
│
├── Diabetes/                          # Diabetes Prediction Project
│   ├── Pima.ipynb                    # Main analysis notebook
│   ├── diabetes.csv                  # PIMA Indians dataset (768 records)
│   ├── README.md                     # Detailed diabetes project documentation
│   └── requirements.txt              # Python dependencies
│
└── Kidney Disease/                    # CKD Detection Project
    ├── CKD_Analysis.ipynb            # Main analysis notebook
    ├── kidney_disease.csv            # Original CKD dataset (400 records)
    ├── ckd_preprocessed_scaled.csv   # Preprocessed & scaled features
    ├── ckd_preprocessed_unscaled.csv # Preprocessed features (unscaled)
    ├── README.md                     # Detailed CKD project documentation
    ├── requirements.txt              # Python dependencies
    │
    ├── model_logistic_regression.pkl  # Trained logistic regression model
    ├── best_model_logistic_regression.pkl # Best tuned model
    ├── model_k-nn.pkl                # k-Nearest Neighbors model
    ├── scaler.pkl                    # StandardScaler object
    │
    ├── model_results.json            # Complete model performance metrics
    └── preprocessing_report.json     # Detailed preprocessing documentation
```

---

## ✨ Key Features

### Data Science Best Practices

- ✅ **Comprehensive EDA** - Distribution analysis, correlation studies, outlier detection
- ✅ **Statistical Rigor** - Hypothesis testing with parametric & non-parametric tests
- ✅ **Missing Data Handling** - Intelligent imputation strategies (median/mode)
- ✅ **Feature Engineering** - Domain-knowledge-driven feature creation
- ✅ **Outlier Treatment** - IQR-based Winsorization
- ✅ **Feature Scaling** - StandardScaler normalization
- ✅ **Class Imbalance** - SMOTE oversampling (Diabetes project)
- ✅ **Cross-Validation** - Stratified k-fold validation (5-fold & 10-fold)
- ✅ **Hyperparameter Tuning** - GridSearchCV optimization
- ✅ **Model Explainability** - SHAP values for interpretability
- ✅ **Reproducibility** - Fixed random seeds, version-controlled code

### Machine Learning Models

| Model | Diabetes | CKD |
|-------|----------|-----|
| **Logistic Regression** | ✅ ROC-AUC: 0.84 | ✅ ROC-AUC: 1.00 |
| **Random Forest** | ✅ ROC-AUC: 0.89 | ❌ |
| **XGBoost** | ✅ ROC-AUC: 0.90 | ❌ |
| **Naive Bayes** | ❌ | ✅ ROC-AUC: 1.00 |
| **k-Nearest Neighbors** | ❌ | ✅ ROC-AUC: 1.00 |

### Visualization Library

- 📊 100+ Professional visualizations across both projects
- 📈 Distribution plots (histograms, KDE, boxplots, violin plots)
- 🔥 Correlation heatmaps
- 📉 ROC & Precision-Recall curves
- 🎯 Confusion matrices
- 📊 Feature importance charts
- 🔍 SHAP summary & dependence plots
- 📊 Partial dependence plots
- 🎨 Publication-ready figures with color gradients

---

## 🛠️ Technologies Used

### Core Libraries

```python
# Data Manipulation
pandas>=1.3.0
numpy>=1.21.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
missingno>=0.5.0

# Statistical Analysis
scipy>=1.7.0
statsmodels>=0.13.0

# Machine Learning
scikit-learn>=1.0.0
xgboost>=1.5.0
imbalanced-learn>=0.9.0

# Model Explainability
shap>=0.40.0

# Development
jupyter>=1.0.0
```

### Development Environment

- **Python**: 3.8+
- **IDE**: Jupyter Notebook
- **Version Control**: Git

---

## 📊 Projects

### 1. Diabetes Prediction (PIMA Indians Dataset)

**Location**: `Diabetes/`

#### Dataset Overview
- **Source**: PIMA Indians Diabetes Database (National Institute of Diabetes)
- **Records**: 768 patients
- **Features**: 8 diagnostic measurements
- **Target**: Binary (Diabetic: 1, Non-diabetic: 0)
- **Population**: Female PIMA Indians, age 21+

#### Key Features
| Feature | Type | Description | Normal Range |
|---------|------|-------------|--------------|
| Pregnancies | Numeric | Number of pregnancies | 0-17 |
| Glucose | Numeric | Plasma glucose (2-hr OGTT) | 70-140 mg/dL |
| BloodPressure | Numeric | Diastolic BP | 60-80 mm Hg |
| SkinThickness | Numeric | Triceps skin fold | 10-50 mm |
| Insulin | Numeric | 2-hour serum insulin | 16-166 mu U/ml |
| BMI | Numeric | Body Mass Index | 18.5-24.9 kg/m² |
| DiabetesPedigreeFunction | Numeric | Genetic influence | 0.08-2.42 |
| Age | Numeric | Age in years | 21-81 |

#### Model Performance

**Best Model: XGBoost**

| Metric | Score |
|--------|-------|
| **Accuracy** | 83.0% |
| **ROC-AUC** | 0.901 |
| **Precision** | 79.3% |
| **Recall** | 82.0% |
| **F1-Score** | 0.806 |

#### Top Predictive Features
1. **Glucose** (30% importance) - Strongest predictor
2. **BMI** (18% importance) - Key metabolic indicator  
3. **Age** (15% importance) - Significant risk factor

#### Statistical Findings
- ✅ **H1**: Diabetic patients have significantly higher glucose levels (+31.68 mg/dL, p<0.001)
- ✅ **H2**: BMI positively correlates with diabetes (ρ=0.314, p<0.001)
- ✅ **H3**: Diabetic patients are significantly older (+5.88 years, p<0.001)

#### Documentation
📖 [Complete Diabetes Project Documentation](Diabetes/README.md)

---

### 2. Chronic Kidney Disease Detection

**Location**: `Kidney Disease/`

#### Dataset Overview
- **Source**: [CKD Dataset (Kaggle)](https://www.kaggle.com/datasets/mansoordaku/ckdisease)
- **Records**: 400 patients
- **Features**: 24 clinical/diagnostic attributes
- **Target**: Binary (CKD: 1, Not CKD: 0)
- **Challenge**: High missing data rate (up to 40% in some features)

#### Key Features

**Numeric Features**:
- age, bp (blood pressure), bgr (blood glucose random)
- bu (blood urea), sc (serum creatinine)
- sod (sodium), pot (potassium)
- hemo (hemoglobin), pcv (packed cell volume)
- wc (white blood cell count), rc (red blood cell count)

**Categorical Features**:
- sg (specific gravity), al (albumin), su (sugar)
- rbc (red blood cells), pc (pus cell), pcc (pus cell clumps)
- ba (bacteria), htn (hypertension), dm (diabetes mellitus)
- cad (coronary artery disease), appet (appetite), pe (pedal edema), ane (anemia)

#### Engineered Features
1. **BUN/Creatinine Ratio** - Kidney function indicator
2. **Albumin-Protein Index** - Protein metabolism marker
3. **Comorbidity Count** - Total disease burden
4. **Risk Score** - Composite clinical risk metric
5. **Age Category** - Life stage binning
6. **Hemo/RBC Ratio** - Anemia severity indicator
7. **BP Category** - Blood pressure classification

#### Model Performance

**Best Model: Logistic Regression (Tuned)**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 100.0% | 100.0% | 100.0% | 100.0% | 1.000 |
| **Naive Bayes** | 95.0% | 100.0% | 92.0% | 95.8% | 1.000 |
| **k-NN** | 98.8% | 100.0% | 98.0% | 99.0% | 0.999 |

#### Best Hyperparameters (Logistic Regression)
- **C**: 1 (regularization strength)
- **Penalty**: L1 (Lasso)
- **Solver**: saga

#### Cross-Validation Results (10-Fold)
- **Logistic Regression**: 0.9997 ± 0.0008
- **Naive Bayes**: 1.0000 ± 0.0000
- **k-NN**: 0.9993 ± 0.0009

#### Statistical Findings
- ✅ **H1**: Hemoglobin levels significantly lower in CKD patients
- ✅ **H2**: Hypertension strongly associated with CKD (Chi-square test)
- ✅ **H3**: Blood urea varies significantly across albumin levels (ANOVA)
- ✅ **H4**: Specific gravity negatively correlated with CKD severity
- ✅ **H5**: Hemoglobin-Creatinine correlation confirms kidney-anemia link

#### Data Preprocessing Highlights

**Missing Value Treatment**:
- Numeric features: Median imputation
- Categorical features: Mode imputation
- High missingness handled: rbc (38%), wc (26%), rc (33%)

**Outlier Treatment**:
- Method: IQR-based capping (Winsorization)
- Features treated: All numeric variables
- Outliers capped without data loss

**Encoding**:
- Binary features: 0/1 encoding
- Ordinal features: Ordered numeric mapping
- Target: CKD=1, Not CKD=0

#### Documentation
📖 [Complete CKD Project Documentation](Kidney%20Disease/README.md)

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Jupyter Notebook
jupyter --version
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/worldisconfusion/Predictive-Modeling-for-Diabetes-and-Chronic-Kidney-Disease-Detection.git
cd Predictive-Modeling-for-Diabetes-and-Chronic-Kidney-Disease-Detection
```

2. **Choose a project to explore**

#### Option A: Diabetes Prediction

```bash
cd Diabetes
pip install -r requirements.txt
jupyter notebook Pima.ipynb
```

#### Option B: Chronic Kidney Disease

```bash
cd "Kidney Disease"
pip install -r requirements.txt
jupyter notebook CKD_Analysis.ipynb
```

### Usage Example

#### Diabetes Prediction

```python
import pickle
import numpy as np

# Load trained model
with open('models/xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

# New patient data
patient = {
    'Pregnancies': 2,
    'Glucose': 140,
    'BloodPressure': 75,
    'SkinThickness': 25,
    'Insulin': 120,
    'BMI': 32.5,
    'DiabetesPedigreeFunction': 0.5,
    'Age': 35
}

# Predict probability
probability = model.predict_proba([list(patient.values())])[0][1]
print(f"Diabetes Risk: {probability:.2%}")
```

#### CKD Detection

```python
import pickle
import pandas as pd

# Load trained model and scaler
with open('Kidney Disease/best_model_logistic_regression.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('Kidney Disease/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# New patient data (simplified example)
patient_data = pd.DataFrame({
    'age': [55], 'bp': [80], 'bgr': [121],
    'bu': [42], 'sc': [1.3], 'hemo': [12.6],
    # ... add all required features
})

# Preprocess and predict
scaled_data = scaler.transform(patient_data)
prediction = model.predict(scaled_data)[0]
probability = model.predict_proba(scaled_data)[0][1]

print(f"CKD Status: {'Positive' if prediction == 1 else 'Negative'}")
print(f"Confidence: {probability:.2%}")
```

---

## 📊 Dataset Information

### Diabetes Dataset (PIMA Indians)

- **Original Source**: Smith et al. (1988), Proceedings of the Annual Symposium on Computer Application in Medical Care
- **Repository**: [UCI ML Repository / Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Size**: 768 records × 9 features
- **Missing Data**: ~48.7% in Insulin, ~29.6% in SkinThickness (zeros represent missing)
- **Class Balance**: 65.1% non-diabetic, 34.9% diabetic (handled with SMOTE)

### CKD Dataset

- **Original Source**: UCI Machine Learning Repository
- **Repository**: [Kaggle CKD Dataset](https://www.kaggle.com/datasets/mansoordaku/ckdisease)
- **Size**: 400 records × 25 features
- **Missing Data**: Up to 40% in some features (handled with imputation)
- **Class Balance**: 62.5% CKD, 37.5% Not CKD (imbalanced but acceptable)

---

## 📈 Model Performance

### Comprehensive Metrics Summary

| Project | Best Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---------|-----------|----------|-----------|--------|----------|---------|
| **Diabetes** | XGBoost | 83.0% | 79.3% | 82.0% | 80.6% | **0.901** |
| **CKD** | Logistic Regression | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **1.000** |

### Performance Notes

**Diabetes Project**:
- Realistic performance on challenging dataset
- ROC-AUC of 0.90 exceeds typical literature benchmarks (0.75-0.88)
- Balanced precision-recall trade-off suitable for clinical screening

**CKD Project**:
- Exceptionally high performance (100% accuracy)
- Likely due to:
  - Smaller dataset with clear patterns
  - Comprehensive feature engineering
  - Optimal preprocessing pipeline
  - Strong clinical markers (hemoglobin, creatinine, etc.)
- ⚠️ **Caution**: Perfect scores suggest careful external validation needed before deployment

---

## 📖 Documentation

### Project-Specific Documentation

Each project contains extensive documentation covering:

1. **Problem Statement & Significance**
2. **Data Description & Collection**
3. **Exploratory Data Analysis**
4. **Preprocessing Pipeline**
5. **Statistical Hypothesis Testing**
6. **Model Building & Comparison**
7. **Visualization & Interpretation**
8. **Conclusions & Insights**
9. **Limitations & Future Work**
10. **Ethical Considerations**
11. **Installation & Usage**
12. **References**

**Diabetes**: [Diabetes/README.md](Diabetes/README.md) - 917 lines of detailed documentation  
**CKD**: [Kidney Disease/README.md](Kidney%20Disease/README.md) - 178 lines with comprehensive analysis

### Additional Resources

**Model Results** (CKD Project):
- `Kidney Disease/model_results.json` - Complete performance metrics
- `Kidney Disease/preprocessing_report.json` - Detailed preprocessing documentation

**Saved Models**:
- Logistic Regression, k-NN models (`.pkl` files)
- StandardScaler for feature scaling (`.pkl`)
- Preprocessed datasets (scaled/unscaled `.csv`)

---

## ⚖️ Ethical Considerations

### ⚠️ IMPORTANT DISCLAIMERS

#### Clinical Use Warning

**These models are NOT**:
- ❌ Replacements for professional medical diagnosis
- ❌ Approved for standalone clinical use
- ❌ Substitutes for standard diagnostic tests
- ❌ FDA/CE-marked medical devices

**These models ARE**:
- ✅ Research prototypes
- ✅ Educational demonstrations
- ✅ Decision support tools (with medical supervision)
- ✅ Risk assessment aids requiring clinical validation

#### Required Actions for Deployment

1. **Clinical Validation**: Must be validated on independent, diverse populations
2. **Regulatory Approval**: Requires FDA clearance or equivalent for clinical use
3. **Professional Supervision**: Must be reviewed by qualified healthcare providers
4. **Standard Diagnostics**: Should complement, not replace, standard tests (HbA1c, OGTT, etc.)

### Data Privacy & Security

- ✅ All datasets are anonymized (no patient identifiers)
- ✅ Aggregate statistics only in public reporting
- ⚠️ HIPAA compliance required for any clinical deployment
- ⚠️ GDPR considerations for international use

### Fairness & Bias Concerns

**Known Limitations**:
1. **Population Bias**: 
   - Diabetes model trained only on PIMA Indian females
   - May not generalize to other ethnicities or males
2. **Selection Bias**: Only patients seeking care are included
3. **Temporal Limitations**: Static, single-timepoint predictions

**Mitigation Strategies**:
- Clearly document training population characteristics
- Report performance by demographic subgroups
- Implement fairness-aware ML techniques
- Regular bias auditing in production

### Transparency & Reproducibility

- ✅ Open-source code (MIT/Educational license)
- ✅ Fixed random seeds for reproducibility
- ✅ Detailed documentation and methodology
- ✅ SHAP explanations for model interpretability
- ✅ Version-controlled development

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Areas for Contribution

1. **Dataset Expansion**
   - Add diverse populations
   - Include longitudinal data
   - Incorporate additional clinical features

2. **Model Improvements**
   - Deep learning implementations
   - Ensemble methods
   - Time-series forecasting

3. **External Validation**
   - Test on independent datasets
   - Multi-center validation studies
   - Performance benchmarking

4. **Deployment Tools**
   - Web application (Flask/Streamlit)
   - REST API development
   - Mobile app integration

5. **Documentation**
   - Translations (non-English)
   - Tutorial videos
   - Clinical use guidelines

### Contribution Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to functions
- Include unit tests for new features
- Update documentation accordingly
- Maintain reproducibility (random seeds)

---

## 📄 License

This project is for **educational and research purposes only**.

- Dataset usage subject to original licensing terms
- Code available under open-source principles
- Not licensed for commercial clinical use without proper validation and regulatory approval

**Disclaimer**: Developers not liable for misuse or medical outcomes. Institutional oversight required for any clinical deployment.

---

## 📚 References

### Datasets

1. **PIMA Indians Diabetes**: Smith, J.W., et al. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proceedings of the Annual Symposium on Computer Application in Medical Care*, 261-265.

2. **CKD Dataset**: UCI Machine Learning Repository, available via [Kaggle](https://www.kaggle.com/datasets/mansoordaku/ckdisease)

### Methodology

3. **SMOTE**: Chawla, N.V., et al. (2002). SMOTE: synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357.

4. **XGBoost**: Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785-794.

5. **SHAP**: Lundberg, S.M., & Lee, S.I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

### Medical Context

6. **Diabetes Guidelines**: American Diabetes Association. (2023). Standards of Medical Care in Diabetes. *Diabetes Care*, 46(Supplement 1).

7. **CKD Guidelines**: KDIGO. (2024). Clinical Practice Guideline for the Evaluation and Management of Chronic Kidney Disease. *Kidney International Supplements*.

8. **ML in Healthcare**: Rajkomar, A., et al. (2019). Machine learning in medicine. *New England Journal of Medicine*, 380(14), 1347-1358.

---

## 🙏 Acknowledgments

- **Dataset Providers**: 
  - National Institute of Diabetes and Digestive and Kidney Diseases (PIMA dataset)
  - UCI Machine Learning Repository (CKD dataset)
- **Communities**: PIMA Indian community, medical research participants
- **Tools**: Scikit-learn, XGBoost, SHAP, Pandas, NumPy, and the open-source Python community
- **Inspiration**: Healthcare professionals working to improve early disease detection

---

## 📧 Contact

For questions, suggestions, or collaborations:

- **GitHub**: [@worldisconfusion](https://github.com/worldisconfusion)
- **Repository**: [Predictive-Modeling-for-Diabetes-and-Chronic-Kidney-Disease-Detection](https://github.com/worldisconfusion/Predictive-Modeling-for-Diabetes-and-Chronic-Kidney-Disease-Detection)

---

## 📊 Project Status

| Component | Status |
|-----------|--------|
| **Diabetes EDA** | ✅ Complete |
| **Diabetes Preprocessing** | ✅ Complete |
| **Diabetes Modeling** | ✅ Complete |
| **Diabetes Documentation** | ✅ Complete |
| **CKD EDA** | ✅ Complete |
| **CKD Preprocessing** | ✅ Complete |
| **CKD Modeling** | ✅ Complete |
| **CKD Documentation** | ✅ Complete |
| **Main Documentation** | ✅ Complete |
| **Model Deployment** | 🔄 Future Work |
| **Web Application** | 🔄 Future Work |
| **External Validation** | 🔄 Future Work |

---

## 🎓 Learning Outcomes

By exploring this repository, you will learn:

1. **Complete ML Pipeline**: From raw data to deployed models
2. **Medical ML Challenges**: Handling missing data, class imbalance, interpretability
3. **Statistical Rigor**: Hypothesis testing, validation strategies
4. **Feature Engineering**: Domain-knowledge-driven feature creation
5. **Model Comparison**: Selecting appropriate algorithms for medical tasks
6. **Explainability**: SHAP values, feature importance, partial dependence
7. **Ethical AI**: Fairness, bias, privacy in healthcare applications
8. **Professional Documentation**: Clear, comprehensive project reporting

---

## 🚀 Future Enhancements

### Short-term (Next 3-6 months)
- [ ] Deploy web application (Streamlit/Flask)
- [ ] Add Partial Dependence Plots (PDPs)
- [ ] Implement fairness analysis
- [ ] Create presentation slides

### Medium-term (6-12 months)
- [ ] Deep learning models (Neural Networks)
- [ ] Ensemble stacking approaches
- [ ] Longitudinal patient tracking
- [ ] External dataset validation

### Long-term (1-2 years)
- [ ] Multi-center clinical trials
- [ ] Regulatory approval process
- [ ] EHR system integration
- [ ] Mobile health application
- [ ] Real-time risk monitoring

---

**Last Updated**: November 2024  
**Version**: 1.0  
**Maintenance Status**: ✅ Active

---

<p align="center">
  <strong>⭐ If you find this project useful, please consider giving it a star! ⭐</strong>
</p>

<p align="center">
  Made with ❤️ for advancing healthcare through data science
</p>
