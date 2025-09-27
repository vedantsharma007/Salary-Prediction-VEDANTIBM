import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import sys


CSV_FILE = 'Salary Data.csv'

try:
    df = pd.read_csv(CSV_FILE, sep='\t')
    print(f"Loaded DataFrame shape: {df.shape}")
except FileNotFoundError:
    print(f"ERROR: '{CSV_FILE}' not found. Please ensure the file is in the VEDANTIBM directory.")
    sys.exit(1)

# === Standardize Column Names ===
# Converts 'Years of Experience' -> 'years_of_experience', etc.
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Define the features and the target
TARGET_COLUMN = 'salary'
NUMERICAL_FEATURES = ['age', 'years_of_experience'] 
CATEGORICAL_FEATURES = ['gender', 'education_level', 'job_title']

# --- 2. Preprocessing ---

# === FIX 1: Drop rows where the target (Salary) is missing ===
# This resolves the "ValueError: Input y contains NaN" crash.
if df[TARGET_COLUMN].isnull().any():
    print(f"Dropping {df[TARGET_COLUMN].isnull().sum()} rows with missing target values...")
    df.dropna(subset=[TARGET_COLUMN], inplace=True)

# === FIX 2: Standardize ALL Categorical VALUES ===
# Converts values like 'Data Scientist' -> 'data_scientist' for consistency
for col in CATEGORICAL_FEATURES:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().str.replace(' ', '_')

# Handle Missing Numerical features (Features: Age, Experience)
for col in NUMERICAL_FEATURES:
    if df[col].isnull().any():
        mean_val = df[col].mean()
        # Use simple assignment to avoid the FutureWarning
        df[col] = df[col].fillna(mean_val) 
        print(f"Filled {col} missing values with mean: {mean_val:.2f}")

# Handle Missing Categorical features
for col in CATEGORICAL_FEATURES:
    if df[col].isnull().any():
        mode_val = df[col].mode()[0]
        # Use simple assignment to avoid the FutureWarning
        df[col] = df[col].fillna(mode_val) 


# One-Hot Encoding
# Now that values are clean, the feature names (e.g., job_title_data_scientist) will be clean too.
df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)

# Define features (X) and target (y)
X = df_encoded.drop(TARGET_COLUMN, axis=1)
y = df_encoded[TARGET_COLUMN]

# Store the list of feature columns for the Flask API
model_features = list(X.columns)
joblib.dump(model_features, 'model_features.pkl') 
print(f"\nFeature list saved successfully. Total Features: {len(model_features)}")

# --- 3. Model Training and Saving ---
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1)
ensemble_reg = VotingRegressor(estimators=[('rf', rf_reg), ('gb', gb_reg)], weights=[1, 1.5], n_jobs=-1)

print("Training the Ensemble Model...")
# Fit on the full data set, which now contains no NaNs in X or y
ensemble_reg.fit(X, y)

# --- 4. Evaluation and Saving ---
model_filename = 'salary_model.pkl'
joblib.dump(ensemble_reg, model_filename)
print(f"\nFinal model saved successfully as {model_filename}")