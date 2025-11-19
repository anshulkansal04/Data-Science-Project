# 🏗️ Project Architecture & Technical Details

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PREDICTIVE MODELING SYSTEM                      │
│          Diabetes & Chronic Kidney Disease Detection                │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
        ┌───────────▼──────────┐    ┌──────────▼───────────┐
        │  DIABETES PROJECT    │    │   CKD PROJECT        │
        │  (PIMA Indians)      │    │   (UCI Dataset)      │
        └───────────┬──────────┘    └──────────┬───────────┘
                    │                           │
        ┌───────────▼──────────┐    ┌──────────▼───────────┐
        │  DATA PIPELINE       │    │  DATA PIPELINE       │
        │  ├─ Load Data        │    │  ├─ Load Data        │
        │  ├─ EDA             │    │  ├─ EDA             │
        │  ├─ Preprocess      │    │  ├─ Preprocess      │
        │  └─ Feature Eng.    │    │  └─ Feature Eng.    │
        └───────────┬──────────┘    └──────────┬───────────┘
                    │                           │
        ┌───────────▼──────────┐    ┌──────────▼───────────┐
        │  MODELING PIPELINE   │    │  MODELING PIPELINE   │
        │  ├─ Train Models     │    │  ├─ Train Models     │
        │  ├─ Validate         │    │  ├─ Validate         │
        │  ├─ Tune Params      │    │  ├─ Tune Params      │
        │  └─ Evaluate         │    │  └─ Evaluate         │
        └───────────┬──────────┘    └──────────┬───────────┘
                    │                           │
        ┌───────────▼──────────┐    ┌──────────▼───────────┐
        │  MODEL ARTIFACTS     │    │  MODEL ARTIFACTS     │
        │  ├─ XGBoost Model    │    │  ├─ Logistic Reg.    │
        │  ├─ Random Forest    │    │  ├─ k-NN Model       │
        │  ├─ Logistic Reg.    │    │  ├─ Naive Bayes      │
        │  └─ SHAP Explainer   │    │  ├─ Scaler Object    │
        └───────────┬──────────┘    │  └─ Results JSON     │
                    │                └──────────┬───────────┘
                    │                           │
        ┌───────────▼──────────┐    ┌──────────▼───────────┐
        │  DOCUMENTATION       │    │  DOCUMENTATION       │
        │  └─ 917-line README  │    │  ├─ 178-line README  │
        └──────────────────────┘    │  ├─ model_results    │
                                     │  └─ preprocess_report│
                                     └──────────────────────┘
```

---

## 🗂️ Data Flow Architecture

### Diabetes Project Data Flow

```
diabetes.csv (768 rows × 9 cols)
    │
    ├─► [1] RAW DATA EXPLORATION
    │       ├─ Statistical summaries
    │       ├─ Distribution analysis
    │       ├─ Missing data patterns
    │       └─ Correlation analysis
    │
    ├─► [2] DATA CLEANING
    │       ├─ Handle zero values (missing data)
    │       ├─ Median imputation (numeric)
    │       ├─ Outlier detection (IQR method)
    │       └─ Winsorization (1-99 percentile)
    │
    ├─► [3] FEATURE ENGINEERING
    │       ├─ BMI categories (4 classes)
    │       ├─ Age groups (4 groups)
    │       ├─ Pregnancy rate (Pregnancies/Age)
    │       ├─ Glucose-BMI ratio
    │       ├─ Log transformations (Insulin, DPF)
    │       └─ 14 total features
    │
    ├─► [4] PREPROCESSING
    │       ├─ StandardScaler normalization
    │       ├─ SMOTE oversampling
    │       │   ├─ Before: 500 vs 268
    │       │   └─ After: 500 vs 500
    │       └─ Train-test split (80/20)
    │
    ├─► [5] MODEL TRAINING
    │       ├─ Logistic Regression
    │       ├─ Random Forest (100 trees)
    │       └─ XGBoost (gradient boosting)
    │
    ├─► [6] MODEL EVALUATION
    │       ├─ 5-Fold Cross-Validation
    │       ├─ ROC-AUC analysis
    │       ├─ Feature importance
    │       ├─ SHAP values
    │       └─ Confusion matrices
    │
    └─► [7] OUTPUTS
            ├─ Trained models (in-memory)
            ├─ 50+ visualizations
            └─ Performance metrics
```

### CKD Project Data Flow

```
kidney_disease.csv (400 rows × 25 cols)
    │
    ├─► [1] RAW DATA EXPLORATION
    │       ├─ Missing data analysis (up to 40%)
    │       ├─ Data type corrections
    │       ├─ Statistical profiling
    │       └─ Target distribution
    │
    ├─► [2] DATA CLEANING
    │       ├─ Missing value imputation
    │       │   ├─ Numeric: median
    │       │   └─ Categorical: mode
    │       ├─ Binary encoding (rbc, pc, pcc, etc.)
    │       ├─ Ordinal encoding (sg: 5 levels)
    │       └─ Target encoding (ckd=1, notckd=0)
    │
    ├─► [3] OUTLIER TREATMENT
    │       ├─ IQR method for all features
    │       ├─ Winsorization (capping)
    │       └─ Detailed report exported
    │
    ├─► [4] FEATURE ENGINEERING
    │       ├─ BUN/Creatinine ratio
    │       ├─ Albumin-Protein index
    │       ├─ Comorbidity count
    │       ├─ Risk score (composite)
    │       ├─ Age categories
    │       ├─ Hemo/RBC ratio
    │       ├─ BP categories
    │       └─ 7 new features created
    │
    ├─► [5] PREPROCESSING
    │       ├─ StandardScaler fitting
    │       ├─ Export unscaled data
    │       ├─ Export scaled data
    │       └─ Save preprocessing report
    │
    ├─► [6] MODEL TRAINING
    │       ├─ Logistic Regression
    │       ├─ Naive Bayes (Gaussian)
    │       └─ k-Nearest Neighbors
    │
    ├─► [7] HYPERPARAMETER TUNING
    │       ├─ GridSearchCV
    │       │   ├─ Logistic: C, penalty, solver
    │       │   └─ k-NN: n_neighbors, weights, metric
    │       └─ Best params saved
    │
    ├─► [8] MODEL VALIDATION
    │       ├─ 5-Fold Cross-Validation
    │       ├─ 10-Fold Cross-Validation
    │       ├─ Bootstrap CI (1000 iterations)
    │       └─ SHAP analysis
    │
    ├─► [9] STATISTICAL TESTING
    │       ├─ H1: Hemoglobin (t-test, Mann-Whitney)
    │       ├─ H2: Hypertension (Chi-square)
    │       ├─ H3: Blood urea by albumin (ANOVA)
    │       ├─ H4: Specific gravity (Spearman)
    │       └─ H5: Hemo-Creatinine (Pearson)
    │
    └─► [10] OUTPUTS
            ├─ model_logistic_regression.pkl
            ├─ best_model_logistic_regression.pkl
            ├─ model_k-nn.pkl
            ├─ scaler.pkl
            ├─ ckd_preprocessed_scaled.csv
            ├─ ckd_preprocessed_unscaled.csv
            ├─ model_results.json
            ├─ preprocessing_report.json
            └─ 100+ visualizations
```

---

## 🧩 Component Architecture

### 1. Data Ingestion Layer

```python
# Component Responsibilities:
# - Load raw CSV data
# - Initial data validation
# - Basic statistics generation

Class: DataLoader
├── load_diabetes_data()
│   └── Returns: pandas.DataFrame (768 × 9)
│
└── load_ckd_data()
    └── Returns: pandas.DataFrame (400 × 25)
```

### 2. Exploratory Data Analysis (EDA) Layer

```python
# Component Responsibilities:
# - Univariate analysis
# - Bivariate analysis
# - Multivariate analysis
# - Visualization generation

Class: EDAAnalyzer
├── analyze_distributions()
├── detect_missing_data()
├── correlation_analysis()
├── outlier_detection()
└── generate_visualizations()
```

### 3. Preprocessing Layer

```python
# Component Responsibilities:
# - Missing value imputation
# - Outlier treatment
# - Feature encoding
# - Feature scaling

Class: DataPreprocessor
├── handle_missing_values()
│   ├── impute_numeric(strategy='median')
│   └── impute_categorical(strategy='mode')
│
├── treat_outliers()
│   └── winsorize(percentiles=(1, 99))
│
├── encode_features()
│   ├── binary_encoding()
│   └── ordinal_encoding()
│
└── scale_features()
    └── StandardScaler.fit_transform()
```

### 4. Feature Engineering Layer

```python
# Component Responsibilities:
# - Create domain-specific features
# - Interaction terms
# - Transformations

Class: FeatureEngineer
├── create_diabetes_features()
│   ├── bmi_categories()
│   ├── age_groups()
│   ├── pregnancy_rate()
│   ├── glucose_bmi_ratio()
│   └── log_transformations()
│
└── create_ckd_features()
    ├── bun_creatinine_ratio()
    ├── albumin_protein_index()
    ├── comorbidity_count()
    ├── risk_score()
    ├── age_categories()
    ├── hemo_rbc_ratio()
    └── bp_categories()
```

### 5. Statistical Testing Layer

```python
# Component Responsibilities:
# - Hypothesis formulation
# - Statistical test execution
# - Result interpretation

Class: StatisticalTester
├── test_group_differences()
│   ├── t_test()
│   ├── mann_whitney_u()
│   └── anova()
│
├── test_correlations()
│   ├── pearson_correlation()
│   └── spearman_correlation()
│
└── test_independence()
    └── chi_square_test()
```

### 6. Model Training Layer

```python
# Component Responsibilities:
# - Model instantiation
# - Training execution
# - Hyperparameter tuning

Class: ModelTrainer
├── train_diabetes_models()
│   ├── LogisticRegression()
│   ├── RandomForestClassifier(n_estimators=100)
│   └── XGBClassifier()
│
└── train_ckd_models()
    ├── LogisticRegression()
    ├── GaussianNB()
    └── KNeighborsClassifier()
```

### 7. Model Evaluation Layer

```python
# Component Responsibilities:
# - Performance metric calculation
# - Cross-validation
# - Model comparison

Class: ModelEvaluator
├── calculate_metrics()
│   ├── accuracy()
│   ├── precision()
│   ├── recall()
│   ├── f1_score()
│   ├── roc_auc_score()
│   ├── sensitivity()
│   └── specificity()
│
├── cross_validate()
│   ├── stratified_kfold(k=5)
│   └── stratified_kfold(k=10)
│
└── compare_models()
    └── generate_comparison_table()
```

### 8. Model Explainability Layer

```python
# Component Responsibilities:
# - Feature importance extraction
# - SHAP value calculation
# - Partial dependence plots

Class: ModelExplainer
├── feature_importance()
│   ├── tree_based_importance()
│   └── permutation_importance()
│
├── shap_analysis()
│   ├── TreeExplainer()
│   ├── summary_plot()
│   └── dependence_plot()
│
└── partial_dependence()
    └── plot_partial_dependence()
```

### 9. Visualization Layer

```python
# Component Responsibilities:
# - Generate publication-ready plots
# - Consistent styling
# - Multi-plot dashboards

Class: Visualizer
├── plot_distributions()
│   ├── histogram()
│   ├── kde_plot()
│   ├── boxplot()
│   └── violin_plot()
│
├── plot_relationships()
│   ├── scatter_plot()
│   ├── heatmap()
│   └── pairplot()
│
├── plot_model_performance()
│   ├── roc_curve()
│   ├── precision_recall_curve()
│   ├── confusion_matrix()
│   └── calibration_curve()
│
└── plot_feature_importance()
    ├── bar_chart()
    └── shap_plots()
```

### 10. Model Persistence Layer

```python
# Component Responsibilities:
# - Save trained models
# - Save preprocessing objects
# - Export results

Class: ModelPersistence
├── save_model(model, filepath)
│   └── pickle.dump()
│
├── save_scaler(scaler, filepath)
│   └── pickle.dump()
│
├── save_results(results, filepath)
│   └── json.dump()
│
└── load_model(filepath)
    └── pickle.load()
```

---

## 🔄 Processing Pipeline

### Diabetes Processing Steps

```
Step 1: Data Loading
  ↓
Step 2: EDA (20+ visualizations)
  ↓
Step 3: Missing Value Analysis
  ↓
Step 4: Imputation (median for numeric)
  ↓
Step 5: Outlier Detection (IQR method)
  ↓
Step 6: Outlier Treatment (Winsorization)
  ↓
Step 7: Feature Engineering (6 new features)
  ↓
Step 8: Feature Scaling (StandardScaler)
  ↓
Step 9: Class Balancing (SMOTE)
  ↓
Step 10: Train-Test Split (80/20, stratified)
  ↓
Step 11: Model Training (3 models)
  ↓
Step 12: Cross-Validation (5-fold)
  ↓
Step 13: Model Evaluation
  ↓
Step 14: Feature Importance Analysis
  ↓
Step 15: SHAP Explainability
  ↓
Step 16: Statistical Testing (3 hypotheses)
  ↓
Step 17: Visualization (50+ plots)
  ↓
Step 18: Documentation & Reporting
```

### CKD Processing Steps

```
Step 1: Data Loading
  ↓
Step 2: Data Quality Assessment
  ↓
Step 3: EDA (35+ visualizations)
  ↓
Step 4: Missing Value Analysis (40% max)
  ↓
Step 5: Missing Value Imputation
  ↓
Step 6: Data Type Correction
  ↓
Step 7: Binary & Ordinal Encoding
  ↓
Step 8: Outlier Detection (IQR method)
  ↓
Step 9: Outlier Treatment (IQR capping)
  ↓
Step 10: Feature Engineering (7 new features)
  ↓
Step 11: Feature Scaling (StandardScaler)
  ↓
Step 12: Export Preprocessed Data (2 versions)
  ↓
Step 13: Export Preprocessing Report (JSON)
  ↓
Step 14: Train-Test Split (80/20, stratified)
  ↓
Step 15: Model Training (3 models)
  ↓
Step 16: Hyperparameter Tuning (GridSearchCV)
  ↓
Step 17: Cross-Validation (5-fold & 10-fold)
  ↓
Step 18: Bootstrap Confidence Intervals
  ↓
Step 19: Model Evaluation & Comparison
  ↓
Step 20: Feature Importance Analysis
  ↓
Step 21: SHAP Explainability
  ↓
Step 22: Statistical Testing (5 hypotheses)
  ↓
Step 23: Advanced Visualization (100+ plots)
  ↓
Step 24: Model Persistence (4 .pkl files)
  ↓
Step 25: Results Export (JSON)
  ↓
Step 26: Documentation & Reporting
```

---

## 🗄️ Data Models

### Diabetes Dataset Schema

```yaml
Dataset: diabetes.csv
Rows: 768
Columns: 9

Features:
  - Pregnancies:
      type: int
      range: [0, 17]
      unit: count
      
  - Glucose:
      type: float
      range: [0, 199]
      unit: mg/dL
      missing_strategy: median_imputation
      
  - BloodPressure:
      type: float
      range: [0, 122]
      unit: mm Hg
      missing_strategy: median_imputation
      
  - SkinThickness:
      type: float
      range: [0, 99]
      unit: mm
      missing_strategy: median_imputation
      missing_rate: 29.6%
      
  - Insulin:
      type: float
      range: [0, 846]
      unit: mu U/ml
      missing_strategy: median_imputation
      missing_rate: 48.7%
      
  - BMI:
      type: float
      range: [0, 67.1]
      unit: kg/m²
      missing_strategy: median_imputation
      
  - DiabetesPedigreeFunction:
      type: float
      range: [0.078, 2.42]
      unit: score
      
  - Age:
      type: int
      range: [21, 81]
      unit: years
      
  - Outcome:
      type: binary
      values: [0, 1]
      labels: ['No Diabetes', 'Diabetes']
      distribution: [65.1%, 34.9%]
```

### CKD Dataset Schema

```yaml
Dataset: kidney_disease.csv
Rows: 400
Columns: 25

Numeric Features:
  - age: {type: float, unit: years, missing: 9}
  - bp: {type: float, unit: mm Hg, missing: 12}
  - bgr: {type: float, unit: mg/dL, missing: 44}
  - bu: {type: float, unit: mg/dL, missing: 19}
  - sc: {type: float, unit: mg/dL, missing: 17}
  - sod: {type: float, unit: mEq/L, missing: 87}
  - pot: {type: float, unit: mEq/L, missing: 88}
  - hemo: {type: float, unit: g/dL, missing: 52}
  - pcv: {type: int, unit: %, missing: 70}
  - wc: {type: int, unit: cells/cumm, missing: 105}
  - rc: {type: float, unit: millions/cmm, missing: 130}

Ordinal Features:
  - sg: {type: ordinal, levels: 5, encoding: [0,1,2,3,4]}
  - al: {type: ordinal, levels: 6, encoding: [0,1,2,3,4,5]}
  - su: {type: ordinal, levels: 6, encoding: [0,1,2,3,4,5]}

Binary Features:
  - rbc: {type: binary, values: [normal, abnormal], encoding: [1, 0]}
  - pc: {type: binary, values: [normal, abnormal], encoding: [1, 0]}
  - pcc: {type: binary, values: [notpresent, present], encoding: [0, 1]}
  - ba: {type: binary, values: [notpresent, present], encoding: [0, 1]}
  - htn: {type: binary, values: [no, yes], encoding: [0, 1]}
  - dm: {type: binary, values: [no, yes], encoding: [0, 1]}
  - cad: {type: binary, values: [no, yes], encoding: [0, 1]}
  - appet: {type: binary, values: [poor, good], encoding: [0, 1]}
  - pe: {type: binary, values: [no, yes], encoding: [0, 1]}
  - ane: {type: binary, values: [no, yes], encoding: [0, 1]}

Target:
  - classification:
      type: binary
      values: [notckd, ckd]
      encoding: [0, 1]
      distribution: [37.5%, 62.5%]
```

---

## 🧮 Algorithm Details

### Diabetes Models

#### 1. Logistic Regression
```python
Parameters:
  - penalty: 'l2'
  - solver: 'lbfgs'
  - max_iter: 1000
  - random_state: 42

Features: 14 (8 original + 6 engineered)
Training samples: 800 (after SMOTE)
Test samples: 200

Performance:
  - ROC-AUC: 0.843
  - Accuracy: 76.5%
```

#### 2. Random Forest
```python
Parameters:
  - n_estimators: 100
  - max_depth: None
  - min_samples_split: 2
  - min_samples_leaf: 1
  - random_state: 42

Features: 14
Training samples: 800
Test samples: 200

Performance:
  - ROC-AUC: 0.888
  - Accuracy: 81.5%
```

#### 3. XGBoost (Best)
```python
Parameters:
  - n_estimators: 100
  - learning_rate: 0.1
  - max_depth: 3
  - random_state: 42

Features: 14
Training samples: 800
Test samples: 200

Performance:
  - ROC-AUC: 0.901
  - Accuracy: 83.0%
```

### CKD Models

#### 1. Logistic Regression (Best)
```python
Initial Parameters:
  - penalty: 'l2'
  - solver: 'lbfgs'
  - max_iter: 1000

Tuned Parameters:
  - C: 1
  - penalty: 'l1'
  - solver: 'saga'

Features: 29 (24 original + 7 engineered - 2 duplicates)
Training samples: 320
Test samples: 80

Performance:
  - ROC-AUC: 1.000
  - Accuracy: 100.0%
```

#### 2. Naive Bayes
```python
Parameters:
  - priors: None (estimated from data)
  - var_smoothing: 1e-09

Features: 29
Training samples: 320
Test samples: 80

Performance:
  - ROC-AUC: 1.000
  - Accuracy: 95.0%
```

#### 3. k-Nearest Neighbors
```python
Initial Parameters:
  - n_neighbors: 5
  - weights: 'uniform'
  - metric: 'euclidean'

Tuned Parameters:
  - n_neighbors: 9
  - weights: 'distance'
  - metric: 'manhattan'

Features: 29
Training samples: 320
Test samples: 80

Performance:
  - ROC-AUC: 0.999
  - Accuracy: 98.8%
```

---

## 📊 Model Artifacts

### Diabetes Project Artifacts (In-Memory Only)

```
No persistent model files
All models stored in notebook kernel memory
Reproducible via notebook re-execution
```

### CKD Project Artifacts

```
Kidney Disease/
├── model_logistic_regression.pkl
│   └── Baseline Logistic Regression (before tuning)
│
├── best_model_logistic_regression.pkl
│   └── Tuned Logistic Regression (C=1, penalty='l1', solver='saga')
│
├── model_k-nn.pkl
│   └── Tuned k-NN (n_neighbors=9, weights='distance', metric='manhattan')
│
├── scaler.pkl
│   └── StandardScaler (fitted on training data)
│       ├── 29 features scaled
│       ├── Mean: stored
│       └── Std: stored
│
├── ckd_preprocessed_scaled.csv
│   └── Fully preprocessed dataset with StandardScaler applied
│
├── ckd_preprocessed_unscaled.csv
│   └── Preprocessed dataset without scaling (for reference)
│
├── model_results.json
│   └── Complete performance metrics
│       ├── Test set results (3 models)
│       ├── Cross-validation results (10-fold)
│       ├── Hyperparameter tuning results
│       └── Best model identification
│
└── preprocessing_report.json
    └── Detailed preprocessing documentation
        ├── Original shape
        ├── Encodings (12 features)
        ├── Missing value strategies (23 features)
        ├── Outlier treatment (24 features)
        ├── Feature engineering (7 features)
        ├── Scaling parameters (29 features)
        └── Class balance metrics
```

---

## 🔐 Security & Privacy Considerations

### Data Protection Mechanisms

```yaml
Anonymization:
  - No patient identifiers in datasets
  - No geographic identifiers
  - Aggregate statistics only

Access Control:
  - Repository: Public (for education)
  - Models: No authentication (prototype)
  - Data: Static files (no live connections)

Compliance Requirements:
  - HIPAA: Required for clinical deployment
  - GDPR: Required for EU deployment
  - FDA: Required for US clinical use
  - IRB: Required for research studies

Recommended Enhancements:
  - Differential privacy for training
  - Federated learning for distributed data
  - Encrypted model storage
  - Audit logging for predictions
  - Access control for APIs
```

---

## 🧪 Testing Strategy

### Current Testing Approach

```yaml
Diabetes Project:
  - Manual validation through notebook execution
  - Visual inspection of outputs
  - Statistical test validation
  - Cross-validation for model reliability

CKD Project:
  - Manual validation through notebook execution
  - Preprocessing report validation
  - Model results JSON validation
  - Cross-validation (5-fold & 10-fold)
  - Bootstrap confidence intervals (1000 iterations)

Limitations:
  - No unit tests
  - No integration tests
  - No continuous integration
  - No automated testing pipeline
```

### Recommended Testing Enhancements

```python
# Unit Tests
tests/
├── test_data_loading.py
├── test_preprocessing.py
├── test_feature_engineering.py
├── test_model_training.py
└── test_model_evaluation.py

# Integration Tests
integration_tests/
├── test_diabetes_pipeline.py
└── test_ckd_pipeline.py

# End-to-End Tests
e2e_tests/
├── test_full_diabetes_workflow.py
└── test_full_ckd_workflow.py
```

---

## 📈 Performance Optimization

### Current Performance Characteristics

```yaml
Diabetes Project:
  Training Time:
    - Logistic Regression: ~1 second
    - Random Forest: ~5 seconds
    - XGBoost: ~10 seconds
  
  Inference Time:
    - Per sample: <1 millisecond
  
  Memory Usage:
    - Dataset: <1 MB
    - Models in memory: ~5 MB

CKD Project:
  Training Time:
    - Logistic Regression: ~0.5 seconds
    - Naive Bayes: ~0.1 seconds
    - k-NN: ~0.2 seconds (+ tuning time)
  
  Inference Time:
    - Per sample: <1 millisecond
  
  Memory Usage:
    - Dataset: <100 KB
    - Models on disk: ~50 KB
    - Preprocessing artifacts: ~100 KB
```

### Optimization Opportunities

```yaml
Data Loading:
  - Current: pandas.read_csv()
  - Optimized: Use chunking for larger datasets
  
Feature Engineering:
  - Current: Sequential computation
  - Optimized: Parallel computation with joblib
  
Model Training:
  - Current: Single-threaded
  - Optimized: Use n_jobs=-1 for tree-based models
  
Hyperparameter Tuning:
  - Current: GridSearchCV (exhaustive)
  - Optimized: RandomizedSearchCV or Bayesian optimization
```

---

## 🔄 Deployment Architecture (Proposed)

### Web Application Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER INTERFACE                       │
│                  (Streamlit / Flask)                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ HTTPS
                     │
┌────────────────────▼────────────────────────────────────┐
│                    API GATEWAY                          │
│              (Authentication & Rate Limiting)           │
└────────────────────┬────────────────────────────────────┘
                     │
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼────────┐        ┌──────▼────────┐
│  DIABETES API  │        │   CKD API     │
│  (Prediction)  │        │  (Prediction) │
└───────┬────────┘        └──────┬────────┘
        │                         │
        │                         │
┌───────▼────────┐        ┌──────▼────────┐
│ Model Serving  │        │ Model Serving │
│  - XGBoost     │        │  - Logistic   │
│  - Scaler      │        │  - Scaler     │
└───────┬────────┘        └──────┬────────┘
        │                         │
        │                         │
┌───────▼─────────────────────────▼────────┐
│           LOGGING & MONITORING            │
│     (Predictions, Performance, Errors)    │
└───────────────────────────────────────────┘
```

---

## 📚 Technology Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Data Storage** | CSV files | Raw & processed data |
| **Data Processing** | pandas, NumPy | Data manipulation |
| **Visualization** | matplotlib, seaborn, missingno | Plotting & charts |
| **Statistics** | SciPy, statsmodels | Hypothesis testing |
| **ML Framework** | scikit-learn | Model training & evaluation |
| **Boosting** | XGBoost | Gradient boosting |
| **Sampling** | imbalanced-learn | SMOTE oversampling |
| **Explainability** | SHAP | Model interpretation |
| **Development** | Jupyter Notebook | Interactive analysis |
| **Version Control** | Git | Code versioning |
| **Serialization** | pickle, JSON | Model & result storage |

---

## 🎯 Key Takeaways

1. **Modular Design**: Each project follows similar architecture but adapts to dataset characteristics
2. **Reproducibility**: Fixed random seeds, detailed documentation, version control
3. **Explainability**: SHAP values and feature importance for model transparency
4. **Clinical Focus**: Medical domain knowledge drives feature engineering
5. **Statistical Rigor**: Hypothesis testing validates medical assumptions
6. **Scalability**: Pipeline design allows easy extension to new diseases
7. **Documentation**: Comprehensive README files and JSON reports
8. **Ethical Considerations**: Privacy, fairness, and clinical disclaimers

---

**Document Version**: 1.0  
**Last Updated**: November 2024  
**Maintainer**: Project Team
