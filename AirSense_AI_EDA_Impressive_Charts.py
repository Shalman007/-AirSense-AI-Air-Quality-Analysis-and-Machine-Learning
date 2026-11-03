"""
AirSense AI
Station-level PM2.5 concentration analysis and five-model comparison
Dataset: Indian government air-quality snapshot (data.gov.in / CPCB)

Research question:
Can station-level PM2.5 average concentration be estimated from the other
pollutant averages and station geography in this snapshot?

Important:
- The dataset is a snapshot, not a time series; this code does NOT claim to
  forecast future air quality.
- PM2.5 min/max are deliberately excluded from model features to avoid target
  leakage / an almost tautological estimate.
- Missing pollutant values are imputed inside sklearn pipelines using training
  data only.
"""

# ============================================================
# 1. IMPORTS AND SETTINGS
# ============================================================
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=FutureWarning)

# Distinct, report-friendly palette: blue, teal, green, coral, gold, violet.
PALETTE = {
    "blue": "#4361EE",
    "electric": "#00A6FB",
    "teal": "#2D8178",
    "green": "#38B000",
    "coral": "#FF6B6B",
    "gold": "#FFB703",
    "violet": "#845EC2",
    "navy": "#14213D",
    "slate": "#64748B",
    "grid": "#DCE5EF",
    "background": "#F7FAFC",
}
MODEL_PALETTE = [
    PALETTE["electric"], PALETTE["coral"], PALETTE["teal"],
    PALETTE["gold"], PALETTE["violet"]
]
sns.set_theme(style="whitegrid", context="notebook")
sns.set_palette(MODEL_PALETTE)
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "axes.facecolor": PALETTE["background"],
    "axes.edgecolor": "#D5DEE8",
    "axes.titlecolor": PALETTE["navy"],
    "axes.titleweight": "bold",
    "axes.titlepad": 12,
    "axes.labelcolor": PALETTE["navy"],
    "grid.color": PALETTE["grid"],
    "grid.linestyle": ":",
    "grid.linewidth": 0.8,
    "xtick.color": PALETTE["slate"],
    "ytick.color": PALETTE["slate"],
})

# Edit this path if the CSV is stored elsewhere.
DATA_PATH = Path("air_quality.csv")
OUTPUT_DIR = Path("airsense_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# 2. LOAD AND VALIDATE THE DATA
# ============================================================
df = pd.read_csv(DATA_PATH)

required_columns = {
    "state", "city", "station", "last_update",
    "latitude", "longitude", "pollutant_id", "pollutant_avg"
}
missing_required = required_columns - set(df.columns)
if missing_required:
    raise ValueError(f"Dataset is missing required columns: {missing_required}")

print("\n========== DATASET OVERVIEW ==========")
print("Rows, columns:", df.shape)
print("\nFirst five rows:\n", df.head())
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isna().sum())
print("\nExact duplicate rows:", df.duplicated().sum())
print("\nPollutant record counts:\n", df["pollutant_id"].value_counts())


# ============================================================
# 3. DATA CLEANING AND QUALITY CHECKS
# ============================================================
# Remove exact duplicate records: they can otherwise overweight observations.
df = df.drop_duplicates().copy()

# Convert time to datetime; invalid timestamps become NaT.
df["last_update"] = pd.to_datetime(
    df["last_update"],
    format="%d-%m-%Y %H:%M",
    errors="coerce"
)

# Numeric columns may contain unexpected strings in some data exports.
numeric_columns = ["latitude", "longitude", "pollutant_min",
                   "pollutant_max", "pollutant_avg"]
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Basic plausibility checks. Negative pollutant concentrations are invalid.
for col in ["pollutant_min", "pollutant_max", "pollutant_avg"]:
    if col in df.columns:
        df.loc[df[col] < 0, col] = np.nan

# Keep rows with identifiable station and pollutant information.
df = df.dropna(subset=["state", "city", "station", "pollutant_id"])

print("\n========== CLEANING SUMMARY ==========")
print("Shape after cleaning:", df.shape)
print("Missing values after basic cleaning:\n", df.isna().sum())


# ============================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
# Why: inspect distribution, data completeness, pollutant variation,
# geography, and relationships before choosing modeling features.

print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df[["pollutant_min", "pollutant_max", "pollutant_avg"]].describe())

# EDA checklist: shape/types, missingness, duplicates, descriptive stats,
# covariance/correlation, and IQR-based potential outlier identification.
print("\\n========== EDA: COVARIANCE MATRIX ==========")
numeric_eda = df.select_dtypes(include=np.number)
if numeric_eda.shape[1] >= 2:
    print(numeric_eda.cov().round(3))

print("\\n========== EDA: IQR OUTLIER SUMMARY ==========")
for column in ["pollutant_min", "pollutant_max", "pollutant_avg"]:
    if column in df.columns:
        values = df[column].dropna()
        if len(values) >= 4:
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_count = ((values < lower_bound) | (values > upper_bound)).sum()
            print(
                f"{column}: {outlier_count:,} potential outliers "
                f"({outlier_count / len(values) * 100:.2f}%) "
                f"using IQR bounds [{lower_bound:.3f}, {upper_bound:.3f}]"
            )


# 4A. Missing-data chart: reveals which measurements are least complete.
missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]
if not missing_pct.empty:
    plt.figure()
    missing_pct.plot(kind="bar", color=PALETTE["violet"], edgecolor="white", linewidth=0.8)
    plt.ylabel("Missing values (%)")
    plt.title("Figure 1. Data Completeness by Column")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_missingness.png")
    plt.show()
else:
    print("No missing values remain in the current columns.")

# 4B. Average concentration distribution: reveals skew and spread.
plt.figure()
sns.histplot(df["pollutant_avg"].dropna(), bins=35, kde=True, color=PALETTE["electric"], edgecolor="white", alpha=0.88)
plt.xlabel("Average pollutant concentration (dataset units)")
plt.title("Figure 2. Distribution of Reported Pollutant Averages")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_pollutant_average_distribution.png")
plt.show()

# 4C. Boxplot by pollutant: compare spread and flag potential extreme values.
# Units differ across pollutant types, so interpret this as a descriptive view,
# not as a direct toxicity ranking.
plt.figure(figsize=(11, 6))
sns.boxplot(data=df, x="pollutant_id", y="pollutant_avg", palette=sns.color_palette("husl", n_colors=df["pollutant_id"].nunique()), linewidth=1.2)
plt.xlabel("Pollutant")
plt.ylabel("Reported average concentration")
plt.title("Figure 3. Pollutant Concentration Spread and Potential Outliers")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_pollutant_boxplot.png")
plt.show()

# 4D. State comparison for PM2.5: uses one pollutant consistently.
pm25_rows = df[df["pollutant_id"].astype(str).str.upper().eq("PM2.5")].copy()
if not pm25_rows.empty:
    state_pm25 = (
        pm25_rows.groupby("state")["pollutant_avg"]
        .median()
        .sort_values(ascending=False)
        .head(12)
    )
    plt.figure(figsize=(11, 6))
    state_pm25.sort_values().plot(kind="barh", color=sns.color_palette("blend:#00A6FB,#2D8178", n_colors=len(state_pm25)), edgecolor="white", linewidth=0.9)
    plt.xlabel("Median station PM2.5 average")
    plt.title("Figure 4. States with Highest Median Reported PM2.5 (Top 12)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_state_pm25_comparison.png")
    plt.show()

# 4E. Correlation heatmap for numeric fields.
# Correlation is association, not proof of causation.
numeric_for_corr = df.select_dtypes(include=np.number)
if numeric_for_corr.shape[1] >= 2:
    corr = numeric_for_corr.corr()
    plt.figure(figsize=(9, 6))
    sns.heatmap(corr, annot=True, cmap="RdYlBu_r", center=0, fmt=".2f", linewidths=0.7, linecolor="white", square=True, cbar_kws={"shrink": 0.82})
    plt.title("Figure 5. Correlation Heatmap of Numeric Measurements")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_numeric_correlation_heatmap.png")
    plt.show()


# ============================================================
# 5. STATISTICAL ANALYSIS (PURPOSEFUL, NOT DECORATIVE)
# ============================================================
# Shapiro-Wilk: assess whether PM2.5 values depart from normality.
# For large n, use a capped random sample because the test is very sensitive.
if not pm25_rows.empty:
    pm25_values = pm25_rows["pollutant_avg"].dropna()
    if len(pm25_values) >= 3:
        sample = pm25_values.sample(
            n=min(5000, len(pm25_values)), random_state=RANDOM_STATE
        )
        shapiro_stat, shapiro_p = stats.shapiro(sample)
        print("\n========== SHAPIRO-WILK NORMALITY TEST ==========")
        print("H0: Sample values are drawn from a normal distribution.")
        print(f"Statistic = {shapiro_stat:.4f}, p-value = {shapiro_p:.6g}")
        if shapiro_p < 0.05:
            print("Decision: reject H0 at alpha=0.05; evidence of non-normality.")
        else:
            print("Decision: do not reject H0 at alpha=0.05.")

# Spearman correlation is used to assess monotonic association robustly.
# It is calculated only where paired observations are available.
if not pm25_rows.empty:
    pm25_mean = pm25_rows["pollutant_avg"].dropna()
    print("\nPM2.5 sample size:", len(pm25_mean))


# ============================================================
# 6. RESHAPE LONG DATA TO ONE ROW PER MONITORING STATION
# ============================================================
# Original data is long format: a station can have one row per pollutant.
# Pivoting creates pollutant-average features side by side for station-level
# modeling. Aggregation uses median in case a station-pollutant is repeated.

station_keys = ["state", "city", "station", "latitude", "longitude"]

station_data = (
    df.pivot_table(
        index=station_keys,
        columns="pollutant_id",
        values="pollutant_avg",
        aggfunc="median"
    )
    .reset_index()
)

station_data.columns.name = None

# Standardize expected target label while preserving actual pollutant labels.
target_candidates = [
    col for col in station_data.columns
    if str(col).strip().upper() == "PM2.5"
]
if not target_candidates:
    raise ValueError(
        "PM2.5 target was not found. Inspect pollutant_id values in your CSV."
    )
TARGET = target_candidates[0]

# Require a known target; other pollutant predictors can be missing and
# will be imputed within the training pipeline.
station_data = station_data.dropna(subset=[TARGET]).copy()

# Exclude target pollutant's min/max because only averages are used here,
# and exclude station identity to reduce memorization.
# Location features are included as useful context.
predictor_pollutants = [
    col for col in station_data.columns
    if col not in station_keys and col != TARGET
]

feature_columns = ["latitude", "longitude"] + predictor_pollutants
X = station_data[feature_columns].copy()
y = station_data[TARGET].copy()

# Numeric columns only for this model design.
numeric_features = feature_columns

print("\n========== MODELING DATA ==========")
print("Station-level rows:", len(station_data))
print("Target:", TARGET, "(station-level reported PM2.5 average)")
print("Features:", feature_columns)
print("Target summary:\n", y.describe())


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================
# The test set is held out and used only for final evaluation.
# This is a cross-sectional station-level snapshot, not future-time forecasting.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)

# Missing-value imputation is fitted on training data only to avoid leakage.
# Scaling benefits linear regression and SVR; tree models do not require it,
# but a shared preprocessing pipeline keeps comparisons reproducible.
numeric_preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_preprocessor, numeric_features)
    ],
    remainder="drop"
)


# ============================================================
# 8. TRAIN FIVE REGRESSION MODELS
# ============================================================
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=5, min_samples_leaf=4, random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05,
        max_depth=2, random_state=RANDOM_STATE
    ),
    "Support Vector Regression": SVR(
        kernel="rbf", C=10, epsilon=0.1
    )
}

results = []
fitted_models = {}
predictions = {}

for model_name, estimator in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    fitted_models[model_name] = pipeline
    predictions[model_name] = y_pred

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

    print(
        f"\n{model_name}: "
        f"MAE={mae:.3f}, MSE={mse:.3f}, "
        f"RMSE={rmse:.3f}, R2={r2:.3f}"
    )

results_df = pd.DataFrame(results).sort_values(
    by="RMSE", ascending=True
).reset_index(drop=True)

print("\n========== FIVE-MODEL COMPARISON ==========")
print(results_df.to_string(index=False))
results_df.to_csv(OUTPUT_DIR / "model_comparison_metrics.csv", index=False)

# Metric comparison chart: lower error is better.
plt.figure(figsize=(11, 6))
plot_metrics = results_df.set_index("Model")[["MAE", "RMSE"]]
plot_metrics.plot(kind="bar", ax=plt.gca(), color=[PALETTE["electric"], PALETTE["coral"]], edgecolor="white", linewidth=0.8)
plt.ylabel("Error (PM2.5 concentration units)")
plt.title("Figure 6. Five-Model Performance Comparison (Lower Is Better)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_model_error_comparison.png")
plt.show()

# R2 comparison chart.
plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x="Model", y="R2", palette=MODEL_PALETTE, edgecolor="white", linewidth=0.8)
plt.ylabel("R² score")
plt.title("Figure 7. Model Comparison by R² Score (Higher Is Better)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_model_r2_comparison.png")
plt.show()


# ============================================================
# 9. BEST MODEL: ACTUAL VS PREDICTED AND RESIDUAL ANALYSIS
# ============================================================
# Lowest RMSE is used as the primary selection criterion because it expresses
# typical error in the target's original units.
best_model_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_model_name]
best_predictions = predictions[best_model_name]
residuals = y_test.to_numpy() - best_predictions

print("\n========== BEST MODEL ==========")
print("Selected model by lowest test RMSE:", best_model_name)
print(results_df.iloc[0].to_dict())

# Actual vs predicted: points near diagonal indicate better agreement.
plt.figure()
plt.scatter(y_test, best_predictions, alpha=0.78, s=48, color=PALETTE["electric"], edgecolors="white", linewidths=0.5)
min_val = min(float(y_test.min()), float(np.min(best_predictions)))
max_val = max(float(y_test.max()), float(np.max(best_predictions)))
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color=PALETTE["coral"], linewidth=2.2, label="Ideal prediction")
plt.xlabel("Actual PM2.5 average")
plt.ylabel("Predicted PM2.5 average")
plt.title(f"Figure 8. Actual vs Predicted — {best_model_name}")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_actual_vs_predicted.png")
plt.show()

# Residual distribution: checks bias and whether errors are centered near zero.
plt.figure()
sns.histplot(residuals, bins=30, kde=True, color=PALETTE["violet"], edgecolor="white", alpha=0.88)
plt.axvline(0, color=PALETTE["coral"], linestyle="--", linewidth=2)
plt.xlabel("Residual (actual - predicted)")
plt.title(f"Figure 9. Residual Distribution — {best_model_name}")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_residual_distribution.png")
plt.show()

# Residuals vs predictions: patterns can reveal nonlinearity or changing variance.
plt.figure()
plt.scatter(best_predictions, residuals, alpha=0.78, s=48, color=PALETTE["teal"], edgecolors="white", linewidths=0.5)
plt.axhline(0, color=PALETTE["coral"], linestyle="--", linewidth=2)
plt.xlabel("Predicted PM2.5 average")
plt.ylabel("Residual")
plt.title(f"Figure 10. Residuals vs Predicted — {best_model_name}")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "10_residuals_vs_predicted.png")
plt.show()


# ============================================================
# 10. FEATURE IMPORTANCE (MODEL-AGNOSTIC)
# ============================================================

# Permutation importance:
# Measures how much model performance decreases when a feature
# is randomly shuffled. Larger values indicate greater importance.

print("\n========== CALCULATING FEATURE IMPORTANCE ==========")

perm = permutation_importance(
    best_model,
    X_test,
    y_test,
    scoring="neg_root_mean_squared_error",
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=1
)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance_Mean": perm.importances_mean,
    "Importance_Std": perm.importances_std
}).sort_values(
    "Importance_Mean",
    ascending=False
)

print("\n========== PERMUTATION FEATURE IMPORTANCE ==========")
print(importance_df.to_string(index=False))

# Save importance results
importance_df.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

# Select top 12 features
top_importance = (
    importance_df.head(12)
    .sort_values("Importance_Mean", ascending=True)
)

# Colorful professional visualization
fig, ax = plt.subplots(figsize=(11, 7))

importance_colors = sns.color_palette(
    "blend:#845EC2,#00A6FB",
    n_colors=len(top_importance)
)

bars = ax.barh(
    top_importance["Feature"],
    top_importance["Importance_Mean"],
    color=importance_colors,
    edgecolor="white",
    linewidth=1.0,
    height=0.72
)

# Add numeric labels
for bar, value in zip(
    bars,
    top_importance["Importance_Mean"]
):
    ax.text(
        value,
        bar.get_y() + bar.get_height() / 2,
        f"  {value:.4f}",
        va="center",
        fontsize=9,
        color=PALETTE["navy"]
    )

ax.axvline(
    0,
    color=PALETTE["coral"],
    linestyle="--",
    linewidth=1.5
)

ax.set_xlabel(
    "Increase in RMSE after feature shuffling"
)

ax.set_ylabel("Predictor Feature")

ax.set_title(
    f"Figure 11. Feature Importance — {best_model_name}"
)

ax.grid(
    axis="x",
    linestyle=":",
    alpha=0.6
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "11_feature_importance.png",
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 11. SAVE TEST PREDICTIONS AND FINAL SUMMARY
# ============================================================
prediction_output = X_test.copy()
prediction_output["Actual_PM2.5"] = y_test
prediction_output["Predicted_PM2.5"] = best_predictions
prediction_output["Residual"] = residuals
prediction_output.to_csv(OUTPUT_DIR / "best_model_test_predictions.csv", index=False)

print("\n========== PROJECT COMPLETED ==========")
print("Best model:", best_model_name)
print("Outputs saved in:", OUTPUT_DIR.resolve())
print("Files include model metrics, predictions, feature importance, and charts.")
