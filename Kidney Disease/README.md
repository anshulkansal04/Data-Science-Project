# 🧬 Chronic Kidney Disease (CKD) Prediction Project

## 📋 Project Overview

This project aims to predict whether a patient suffers from Chronic Kidney Disease based on diagnostic, clinical, and demographic attributes using data science and machine learning techniques.

## 🎯 Problem Statement

To predict whether a patient suffers from Chronic Kidney Disease based on diagnostic, clinical, and demographic attributes such as blood pressure, blood glucose, albumin, hemoglobin, etc.

## 📊 Dataset Information

- **Source**: [CKD Dataset from Kaggle](https://www.kaggle.com/datasets/mansoordaku/ckdisease)
- **Rows**: ~400 patient records
- **Columns**: 25 features (numeric and categorical)
- **Target**: `classification` (ckd / notckd)

### Features Include:
- **Numeric**: age, bp, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc
- **Categorical**: sg, al, su, rbc, pc, pcc, ba, htn, dm, cad, appet, pe, ane

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook

### Installation

1. Clone or download this project
2. Navigate to the project directory:
   ```bash
   cd "Kidney Disease"
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Analysis

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open `CKD_Analysis.ipynb` in your browser

3. Run the cells sequentially to perform the complete analysis

## 📁 Project Structure

```
Kidney Disease/
├── kidney_disease.csv          # Dataset
├── CKD_Analysis.ipynb          # Main analysis notebook
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔍 Analysis Steps

### Step 1: Data Collection & Description
- Load and explore the dataset
- Understand feature types and descriptions
- Basic statistical summaries
- Target variable distribution

### Step 2: Exploratory Data Analysis (EDA)
- **Data Quality Assessment**: Check for duplicates, clean data
- **Missing Value Analysis**: Identify and visualize missing data patterns
- **Univariate Analysis**: Distribution of individual features
- **Bivariate Analysis**: Relationship between features and target
- **Multivariate Analysis**: Complex relationships and interactions
- **Outlier Detection**: Identify and analyze outliers using IQR method

### Step 3: Data Preprocessing ✅
- **Missing Data Handling**: Median/mode imputation
- **Data Type Correction**: Binary and ordinal encoding
- **Outlier Treatment**: IQR-based capping (Winsorization)
- **Feature Engineering**: Created 7 new meaningful features
- **Feature Scaling**: StandardScaler for model readiness
- **Class Balance Check**: Analyzed and strategized
- **Data Export**: Two versions (scaled/unscaled) + detailed report

### Step 4: Hypotheses Formulation & Statistical Testing ✅
- **H1**: Hemoglobin levels comparison (t-test, Mann-Whitney U)
- **H2**: Hypertension association (Chi-square test)
- **H3**: Blood urea across albumin levels (ANOVA, Kruskal-Wallis)
- **H4**: Specific gravity correlation (Spearman rank)
- **H5**: Hemoglobin-Creatinine correlation (Pearson)
- **10 Visualizations**: Professional plots for each hypothesis
- **Statistical Rigor**: Both parametric and non-parametric tests

### Step 5: Analysis, Modeling & Validation ✅
- **4 Models Built**: Logistic Regression, XGBoost, Naive Bayes, k-NN
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, Sensitivity, Specificity
- **Cross-Validation**: 5-Fold and 10-Fold stratified CV
- **Hyperparameter Tuning**: GridSearchCV for optimal parameters
- **Feature Importance**: XGBoost importance & Logistic coefficients
- **SHAP Analysis**: Model explainability with SHAP values
- **Bootstrap CI**: 95% confidence intervals (1000 iterations)
- **20+ Visualizations**: Confusion matrices, ROC curves, feature importance
- **Model Saving**: Best model and all tuned models saved (.pkl files)

### Step 6: Advanced Visualization Plan ✨ FINAL STEP!
- **Correlation Heatmap**: Comprehensive feature relationships
- **Missing Data Viz**: 4-panel missing data analysis
- **Clinical Distributions**: 10+ boxplots by CKD status
- **Categorical Analysis**: 10+ countplots for binary features
- **Comorbidity Charts**: 6 visualizations showing disease relationships
- **Disease Severity**: Scatterplots of key biomarkers
- **Performance Dashboard**: Single-view model comparison
- **35+ New Visualizations**: Publication-ready charts
- **100+ Total Visualizations**: Complete visual story of the analysis

## 📈 Key Findings

The analysis provides insights into:
- Data quality issues and missing value patterns
- Feature distributions and statistical properties
- Significant features correlated with CKD
- Class imbalance in the target variable
- Outliers and extreme values requiring treatment

## 🔄 Project Status

~~1. Data Collection & Description~~ ✅ COMPLETED  
~~2. Exploratory Data Analysis~~ ✅ COMPLETED  
~~3. Data Preprocessing~~ ✅ COMPLETED  
~~4. Hypothesis Testing~~ ✅ COMPLETED  
~~5. Model Building & Validation~~ ✅ COMPLETED  
~~6. Advanced Visualizations~~ ✅ COMPLETED

**🎉 ALL CORE STEPS COMPLETED! 🎉**

**Optional Future Enhancements**:
7. **Model Deployment**: Create a web application for CKD predictions (Flask/Streamlit)
8. **Advanced Explainability**: Add Partial Dependence Plots (PDP) and ICE plots
9. **Fairness Analysis**: Evaluate model performance across age/gender groups
10. **Documentation**: Final technical report and presentation slides

## 🧰 Technologies Used

- **Python**: Programming language
- **Pandas & NumPy**: Data manipulation
- **Matplotlib & Seaborn**: Data visualization
- **Missingno**: Missing data visualization
- **SciPy**: Statistical analysis
- **Jupyter Notebook**: Interactive development environment

## 📝 Notes

- This is an educational/research project for understanding CKD prediction
- The dataset contains real patient data, so results should be interpreted carefully
- Medical decisions should not be based solely on these predictions without professional consultation

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- Dataset source: [Kaggle CKD Dataset](https://www.kaggle.com/datasets/mansoordaku/ckdisease)
- UCI Machine Learning Repository for the original CKD dataset

---

**Author**: Data Science Project  
**Date**: November 2024  
**Version**: 1.0

