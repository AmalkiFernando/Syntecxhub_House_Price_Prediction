# 🏠 House Price Prediction — Linear Regression

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📌 Project Overview

This project uses the **California Housing Dataset** to build a **Linear Regression Model** that predicts median house values based on geographic, demographic, and economic features.

The pipeline includes data cleaning, feature engineering, polynomial feature expansion, and model evaluation — all designed to handle real-world data issues like skewed distributions and capped values.

---

## 🎯 Objectives

1. Load, clean and explore the housing dataset
2. Engineer meaningful features and encode categoricals
3. Train a Linear Regression model with polynomial interactions
4. Evaluate using RMSE, MAE, and R²
5. Interpret model coefficients
6. Save and reload the model for future predictions

---

## 📊 Dataset

- **Source:** California Housing Dataset
- **File:** `housing.csv`
- **Rows:** 20,640
- **Columns:** 10
- **Target:** `median_house_value`

| Feature | Description |
|---|---|
| `longitude` / `latitude` | Geographic coordinates |
| `housing_median_age` | Median age of houses in the district |
| `total_rooms` | Total rooms in the district |
| `total_bedrooms` | Total bedrooms in the district |
| `population` | District population |
| `households` | Number of households |
| `median_income` | Median income (in $10,000s) |
| `ocean_proximity` | Categorical — distance to ocean |
| `median_house_value` | **Target variable** |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Programming language |
| Pandas | Data loading, cleaning, manipulation |
| NumPy | Numerical operations, log transformation |
| Matplotlib / Seaborn | Visualization |
| Scikit-learn | Preprocessing, model training, evaluation |
| Joblib | Model serialization |
| Google Colab | Development environment |

---

## 🤖 Model Used

### Linear Regression (with Polynomial Features)

Plain Linear Regression was used as required, enhanced with **degree-2 interaction terms** via `PolynomialFeatures` to capture non-linear relationships while keeping the model family linear.

---

## ⚙️ ML Pipeline

### 1. Data Cleaning
- Filled missing with median
- Removed capped house values above $500,000 to reduce bias

### 2. Feature Engineering (Engineer 6 new features)
| New Feature | Formula |
|---|---|
| `rooms_per_household` | total_rooms / households |
| `bedrooms_per_room` | total_bedrooms / total_rooms |
| `population_per_household` | population / households |
| `rooms_per_person` | total_rooms / population |
| `bedrooms_per_person` | total_bedrooms / population |
| `income_per_room` | median_income / total_rooms |

### 3. Encoding
- `ocean_proximity` one-hot encoded 

### 4. Target Transformation
- Applied `np.log1p()` to `median_house_value` to fix right skew
- Predictions converted back using `np.expm1()`

### 5. Scaling & Polynomial Features
- `StandardScaler` — fit on train, transform on both
- `PolynomialFeatures(degree=2, interaction_only=True)` — captures feature interaction effects

### 6. Model
- Fit `LinearRegression`

---

## 📈 Model Evaluation

| Metric | Description | Results |
|---|---|---|
| **RMSE** | Root Mean Squared Error (in original $ scale) | $57,526 |
| **MAE** | Mean Absolute Error (in original $ scale) | $38,926 |
| **R²** | Proportion of variance explained by the model | 0.6681

Visualizations included:
- Feature correlation heatmap
- Log-transformed target distribution
- Residuals vs Predicted plot
- Actual vs Predicted scatter plot
- Top 20 coefficient bar chart

---

## 📁 Project Structure
```
House_Pricing_Predict/
│
├── housing.csv                  # Raw dataset
├── House_Pricing_Predict.ipynb  # Main notebook
├── house_price_model.pkl        # Trained model
├── scaler.pkl                   # StandardScaler
├── poly.pkl                     # PolynomialFeatures transformer
└── README.md                    # Project documentation

---

## 👤 Author

**Amalki Fernando**  