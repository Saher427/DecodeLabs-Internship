# 🚀 DecodeLabs Data Science Internship — Project Portfolio
### Saher Irfan | Data Science Intern | Batch 2026

## 👩‍💻 About This Repository

This repository contains all projects completed during my **Data Science Internship at Decode Labs** (June 20 - July 20, 2026). Each project covers a core pillar of Data Science - from raw data cleaning to machine learning model deployment - built using real-world datasets and industry-standard tools.

## 📁 Repository Structure

```
DecodeLabs-Internship/
│
├── Project-1/
│   ├── Project-1.ipynb
│   ├── Dataset_for_Data_Analytics.xlsx
│   └── ecommerce_clean.csv
│
├── Project-2/
│   ├── Project-2.ipynb
│   └── creditcard.csv
│
├── Project-3/
│   ├── Project-3.ipynb
│   ├── Mall_Customers.csv
│   └── customers_segmented.csv
│
└── README.md
```

## 📊 PROJECT 1 - Advanced EDA & Feature Engineering

### 🎯 Goal
Transform raw, chaotic e-commerce data into a mathematically clean dataset ready for machine learning algorithms.

### 📂 Dataset
**E-Commerce Orders Dataset**
- 1,200 rows × 14 columns
- Source: Internal training dataset (DecodeLabs)
- Features: OrderID, Date, CustomerID, Product, Quantity, UnitPrice, PaymentMethod, OrderStatus, CouponCode, ReferralSource, TotalPrice

### 🔧 What Was Done

**Phase 1 - Securing Input Fidelity:**
- Analyzed missing values using the Decision Matrix
- `CouponCode` had 25.75% missing (309 rows) → filled with `"NoCoupon"`
- Dropped irrelevant columns: OrderID, CustomerID, TrackingNumber, ShippingAddress
- Detected outliers using IQR method on UnitPrice, TotalPrice, ItemsInCart, Quantity
- Neutralized outliers using Winsorization (`numpy.clip`) - preserved all 1,200 rows

**Phase 2 - Vectorized Computation Engine:**
- Extracted date features from Date column (Month, DayOfWeek, Year)
- Applied One-Hot Encoding to Product, PaymentMethod, OrderStatus, ReferralSource, CouponCode
- Checked multicollinearity using Pearson correlation heatmap
- No features exceeded 0.80 threshold - all retained

**Phase 3 - Feature Engineering (5 New Features):**
| Feature | Formula | Business Meaning |
|---------|---------|-----------------|
| RevenuePerItem | TotalPrice / Quantity | Average value per unit sold |
| CartConversionRatio | Quantity / ItemsInCart | How much of cart was purchased |
| IsHighValueOrder | TotalPrice > Median | Flag premium orders |
| IsWeekendOrder | DayOfWeek >= 5 | Weekend buying behavior |

**Phase 4 - Structural Validation:**
- Validated final dataset using Pandera schema contracts
- All data type and range checks passed ✅

### 🛠️ Tools & Libraries
`Python` `Pandas` `NumPy` `Matplotlib` `Seaborn` `Pandera` `Scikit-Learn`

### ✅ Key Results
- Dataset cleaned from 14 → 25+ engineered columns
- Zero missing values in final output
- 4 new predictive features engineered
- Pandera validation: **PASSED** ✅
- Output saved as `ecommerce_clean.csv`

## 🔍 PROJECT 2 - Supervised Learning: Fraud Detection Pipeline

### 🎯 Goal
Build and tune a classification pipeline to identify fraudulent credit card transactions in a highly imbalanced dataset.

### 📂 Dataset
**Credit Card Fraud Detection (Kaggle)**
- 284,807 transactions × 31 columns
- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Features: Time, V1-V28 (PCA-anonymized), Amount, Class
- Class Distribution: 284,315 Legitimate (99.83%) vs 492 Fraud (0.17%)

### 🔧 What Was Done

**The Core Problem - Accuracy Trap:**
A model predicting "Legitimate" every time achieves 99.83% accuracy but catches ZERO fraud. This is why Accuracy was completely discarded in favor of Recall and ROC-AUC.

**Pipeline Architecture:**
- Stratified 80/20 Train/Test Split (BEFORE SMOTE — zero data leakage)
- Used `imblearn.pipeline.Pipeline` (NOT sklearn — supports SMOTE natively)

**Model 1 - Logistic Regression Pipeline:**
```
StandardScaler → SMOTE → LogisticRegression
```

**Model 2 - Random Forest Pipeline:**
```
SMOTE → RandomForestClassifier
(No scaler needed - tree-based models are scale-invariant)
```

**Hyperparameter Tuning:**
- GridSearchCV with StratifiedKFold (3-fold)
- Optimized for `recall` scoring
- LR best C: 0.01 | RF best depth: 20

### 📊 Final Results

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Logistic Regression | 0.0585 | 0.9184 | 0.1100 | 0.9715 |
| **Random Forest** | **0.5148** | **0.8878** | **0.6517** | **0.9853** |

**🏆 Winner: Random Forest (ROC-AUC: 0.9853)**

**Confusion Matrix (Random Forest):**
- True Positives (Fraud Caught): 87
- False Negatives (Fraud Missed): 11
- False Positives (False Alarms): 82

**Top 3 Fraud Indicators (Feature Importance):**
1. V14: 0.3090
2. V10: 0.1257
3. V4: 0.1059

### 🛠️ Tools & Libraries
`Python` `Pandas` `NumPy` `Scikit-Learn` `Imbalanced-Learn` `Matplotlib` `Seaborn`

### ✅ Key Results
- Both models achieved ROC-AUC well above 0.85 industry target ✅
- Random Forest: ROC-AUC **0.9853** | Recall **88.78%**
- Logistic Regression: ROC-AUC **0.9715** | Recall **91.84%**
- Zero data leakage - SMOTE applied strictly inside pipeline

## 👥 PROJECT 3 - Unsupervised Learning: Customer Segmentation

### 🎯 Goal
Use distance-based algorithms to discover hidden mathematical groupings in unlabeled retail customer data and translate them into actionable business personas.

### 📂 Dataset
**Mall Customer Segmentation Dataset (Kaggle)**
- 200 customers × 5 columns
- Source: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
- Features: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100)

### 🔧 What Was Done

**The IPO Pipeline Architecture:**

**Phase 1 - SCALE:**
- Encoded Gender (Female=1, Male=0)
- Applied StandardScaler to all 4 features
- Prevented Annual Income ($0-$137k) from dominating Spending Score (1-100)
- After scaling: mean ≈ 0, std ≈ 1 for all features ✅

**Phase 2 - COMPRESS (PCA):**
- Applied PCA to find optimal components
- 4 components needed to retain 95% variance
- PC1: 33.8% | PC2: 26.1% | PC3: 23.2% | PC4: 16.7%
- 2D version created for visualization

**Phase 3 - CLUSTER (K-Means):**
- Elbow Method run for K=1 to K=10
- Silhouette Score calculated for K=2 to K=10
- Final K=5 selected based on visual elbow analysis and business logic
- Silhouette Score at K=5: **0.3041** (acceptable for real customer data)

**Phase 4 - TRANSLATE (Business Personas):**

| Cluster | Persona | Avg Income | Avg Spending | Avg Age | Strategy |
|---------|---------|-----------|-------------|---------|---------|
| 0 🔴 | High-Value Trendsetters | $86k | 82 | 33 | Exclusive perks, premium loyalty |
| 1 🔵 | Affluent Conservatives | $88k | 19 | 36 | Trust content, warranties |
| 2 🟢 | Mid-Tier Regulars | $49k | 40 | 50 | Cross-selling, moderate offers |
| 3 🟡 | Budget-Conscious Explorers | $40k | 61 | 26 | Flash sales, BNPL, influencers |
| 4 🟣 | Conservative Minimizers | $54k | 37 | 54 | Basic utility, clear value |

### 🛠️ Tools & Libraries
`Python` `Pandas` `NumPy` `Scikit-Learn` `Matplotlib` `Seaborn`

### ✅ Key Results
- 5 mathematically distinct customer segments identified
- PCA reduced 4 features retaining 95% variance
- Each cluster translated into a clear business persona and marketing strategy
- Centroids reverse-engineered back to human-readable metrics
- Output saved as `customers_segmented.csv`

## 🧰 Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | Scikit-Learn, Imbalanced-Learn |
| Visualization | Matplotlib, Seaborn |
| Data Validation | Pandera |
| Environment | Jupyter Notebook |
| Version Control | Git & GitHub |

## 📈 Skills Demonstrated

| Skill | Project |
|-------|---------|
| Missing Value Imputation | Project 1 |
| Outlier Detection & Winsorization | Project 1 |
| Feature Engineering | Project 1 |
| Data Validation (Pandera) | Project 1 |
| Class Imbalance (SMOTE) | Project 2 |
| Classification Models | Project 2 |
| Hyperparameter Tuning (GridSearchCV) | Project 2 |
| Precision / Recall / ROC-AUC Evaluation | Project 2 |
| Dimensionality Reduction (PCA) | Project 3 |
| K-Means Clustering | Project 3 |
| Elbow Method + Silhouette Score | Project 3 |
| Business Intelligence Translation | Project 3 |

## 🏢 About the Internship

| Detail | Info |
|--------|------|
| Company | Decode Labs |
| Position | Data Science Intern |
| Duration | June 20 – July 20, 2026 |
| Mode | Remote / Virtual |
| Website | www.decodelabs.tech |

## 📬 Contact

**Saher Irfan**
📧 saherirfan07@gmail.com
🔗 [LinkedIn Profile]
🐙 [GitHub Profile]

*This portfolio was built as part of the DecodeLabs Industrial Training Kit - Batch 2026*