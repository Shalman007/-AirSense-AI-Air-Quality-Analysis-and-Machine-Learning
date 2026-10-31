# 🌍 AirSense AI – Air Quality Analysis and Machine Learning

## 📌 Project Overview

**AirSense AI** is a Data Science and Machine Learning project that analyzes air quality data from India to understand pollutant patterns and estimate PM2.5 concentration.

The project applies data preprocessing, exploratory data analysis (EDA), statistical analysis, visualization, and multiple machine learning algorithms. The performance of each model is evaluated and compared using regression metrics.

> **Project scope:** This project estimates PM2.5 concentration using station-level pollutant and geographic information from a dataset snapshot. It does not perform future-time air quality forecasting.

---

## 🎯 Objectives

- Analyze air quality measurements across locations in India.
- Clean and preprocess the dataset.
- Identify missing values and potential outliers.
- Explore relationships between different pollutants.
- Examine the correlation and covariance among numerical variables.
- Estimate PM2.5 concentration using machine learning.
- Compare multiple models using standard evaluation metrics.
- Identify influential features using permutation importance.

---

## 📂 Dataset

The dataset contains air quality measurements from monitoring stations in India.

**Dataset dimensions:**
- Initial records: 3,843
- Columns: 11
- Exact duplicate rows: 0
- Missing values: 334 each in `pollutant_min`, `pollutant_max`, and `pollutant_avg`

### Dataset Features

| Feature | Description |
|---|---|
| `country` | Country name |
| `state` | State or region |
| `city` | City name |
| `station` | Monitoring station |
| `last_update` | Latest recorded update |
| `latitude` | Station latitude |
| `longitude` | Station longitude |
| `pollutant_id` | Pollutant identifier |
| `pollutant_min` | Minimum pollutant measurement |
| `pollutant_max` | Maximum pollutant measurement |
| `pollutant_avg` | Average pollutant measurement |

The dataset includes pollutant measurements such as PM2.5, PM10, NO2, CO, OZONE, SO2, and NH3.

---

## 🧹 Data Preprocessing

The preprocessing workflow includes:

1. Inspecting dataset structure and data types.
2. Checking for missing values.
3. Checking and removing duplicate records, if present.
4. Handling missing pollutant measurements.
5. Preparing station-level data for modeling.
6. Selecting relevant numerical features.
7. Separating input features and target variable.

### Target Variable

`PM2.5` average concentration.

### Features Used

- Latitude
- Longitude
- Average CO
- Average NH3
- Average NO2
- Average OZONE
- Average PM10
- Average SO2

PM2.5 minimum and maximum values were excluded from the input features to reduce target leakage.

---

## 📊 Exploratory Data Analysis (EDA)

The project explores air quality data through:

- Pollutant average distribution
- Pollutant boxplots
- State-wise PM2.5 comparison
- Correlation heatmap
- Missing-value analysis
- IQR-based outlier detection
- Statistical summary
- Shapiro-Wilk normality test

The analysis helps identify pollutant distributions, potential outliers, and relationships among air quality variables.

---

## 🤖 Machine Learning Models

Five regression algorithms were trained and compared:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. Support Vector Regression (SVR)

### Evaluation Metrics

| Metric | Meaning |
|---|---|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of Determination |

---

## 🏆 Model Performance

The following results were obtained from the evaluated models:

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 16.371 | 515.174 | 22.697 | 0.633 |
| Decision Tree | 14.856 | 389.946 | 19.747 | 0.722 |
| Random Forest | 14.283 | 386.279 | 19.654 | 0.725 |
| Gradient Boosting | 13.530 | 355.492 | 18.854 | 0.747 |
| SVR | 13.900 | 367.603 | 19.173 | 0.738 |

**Gradient Boosting achieved the best overall performance** among the evaluated models, with the lowest RMSE and highest R² score.

---

## 🔎 Feature Importance

Permutation importance was used to examine the contribution of input features to model performance.

The analysis identified **PM10 average concentration** as the most influential feature for the PM2.5 estimation model.

Feature importance indicates predictive contribution within this model; it does not establish a causal relationship.

---

## 📈 Visualizations

The project includes visualizations such as:

- Pollutant average distribution
- Pollutant boxplots
- State-wise PM2.5 comparison
- Correlation heatmap
- Model MAE and RMSE comparison
- Model R² comparison
- Actual vs. predicted PM2.5
- Residual distribution
- Residuals vs. predicted values
- Permutation feature importance

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

### 2. Navigate to the project directory

```bash
cd YOUR-REPOSITORY
```

### 3. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Run the notebook

```bash
jupyter notebook
```

Open the project notebook and execute the cells in order.

---

## 📌 Key Findings

- The dataset contained missing pollutant measurements that required preprocessing.
- PM2.5 measurements showed a non-normal distribution according to the Shapiro-Wilk test.
- PM10 showed the strongest predictive contribution in the permutation importance analysis.
- Gradient Boosting performed best among the five evaluated regression models.
- The model results represent station-level estimation using the prepared dataset, not future air quality forecasting.

---

## 🔮 Future Scope

- Incorporate time-series air quality data for forecasting.
- Integrate real-time air quality monitoring data.
- Develop an interactive dashboard using Streamlit or Power BI.
- Explore advanced machine learning and deep learning models.
- Deploy the model as a web application.
- Expand the analysis to additional locations and time periods.

---

## 👨‍💻 Author

**Anukula Shalman Raju**  
B.Tech – Data Science

Interested in Data Science, Machine Learning, Python, and Data Analytics.

---

## ⭐ Acknowledgment

This project was developed as part of my learning journey in Data Science and Machine Learning.

If you find this project useful, feel free to ⭐ the repository!
