import pandas as pd
import numpy as np
import sys
import os

# Defaults
workspace = os.path.dirname(os.path.abspath(__file__))
default_input = os.path.join(workspace, '..', 'diabetes_extracted.csv')
default_output = os.path.join(workspace, '..', 'diabetes_cleaned.csv')

input_csv = sys.argv[1] if len(sys.argv) > 1 else default_input
output_csv = sys.argv[2] if len(sys.argv) > 2 else default_output

print(f"Reading: {input_csv}")

df = pd.read_csv(input_csv, low_memory=False)
print(f"Initial shape: {df.shape}")
print("Columns:", list(df.columns))

# Normalize column names
df.columns = [str(c).strip() for c in df.columns]

# 1) Drop ID and No_Pation columns (case-insensitive)
cols_to_drop = [c for c in df.columns if c.lower() in ("id", "no_pation")]
if cols_to_drop:
    print("Dropping columns:", cols_to_drop)
    df = df.drop(columns=cols_to_drop)
else:
    print("No `ID` or `No_Pation` columns found to drop.")

# 2) Inspect and clean `Gender` column
gender_col = next((c for c in df.columns if c.lower() == 'gender'), None)
if gender_col:
    print(f"Found gender column: `{gender_col}`")
    print("Unique values before:")
    print(df[gender_col].astype(str).str.strip().replace('nan', np.nan).value_counts(dropna=False))

    def normalize_gender(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().lower()
        if s in ('f', 'female', 'fem', 'femal', 'fe'):
            return 'F'
        if s in ('m', 'male', 'mal', 'ma', 'mail'):
            return 'M'
        # sometimes values like '0' or '1' appear; try numeric map
        if s in ('0', '1'):
            return 'F' if s == '0' else 'M'
        # fallback: single-letter
        if len(s) == 1:
            if s == 'f':
                return 'F'
            if s == 'm':
                return 'M'
        return s.upper()

    df[gender_col] = df[gender_col].apply(lambda x: normalize_gender(x))
    print("Unique values after normalization:")
    print(df[gender_col].value_counts(dropna=False))
else:
    print("No `Gender` column found.")

# 3) One-hot encode categorical columns except target variable
# Try to detect a target column
lower_cols = [c.lower() for c in df.columns]
possible_targets = ('outcome', 'diabetes', 'diabetes_mellitus', 'target', 'class', 'status')
target_col = None
for c in df.columns:
    if c.lower() in possible_targets:
        target_col = c
        break
if target_col:
    print(f"Detected target column: `{target_col}` (will be excluded from one-hot)")
else:
    print("No obvious target column detected; one-hot encoding all categorical columns.")

# Identify categorical (object / category) columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
# Exclude target if detected
if target_col and target_col in cat_cols:
    cat_cols = [c for c in cat_cols if c != target_col]

print("Categorical columns to one-hot encode:", cat_cols)
if cat_cols:
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)

# 4) Handle missing values: numeric -> mean, non-numeric -> mode
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols:
    if df[c].isna().any():
        mean_val = df[c].mean()
        df[c] = df[c].fillna(mean_val)
        print(f"Filled NA in numeric column `{c}` with mean={mean_val}")

# For remaining object columns (should be none after get_dummies), fill mode
obj_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
# Do not fill missing values for the target column; leave them so we can drop unlabeled rows before training
if target_col and target_col in obj_cols:
    obj_cols = [c for c in obj_cols if c != target_col]

for c in obj_cols:
    if df[c].isna().any():
        mode_val = df[c].mode(dropna=True)
        if not mode_val.empty:
            df[c] = df[c].fillna(mode_val.iloc[0])
            print(f"Filled NA in categorical column `{c}` with mode={mode_val.iloc[0]}")
        else:
            df[c] = df[c].fillna('')
            print(f"Filled NA in categorical column `{c}` with empty string")

# Summary
print(f"Final shape: {df.shape}")

# Save cleaned dataset
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False)
print(f"Saved cleaned data to: {output_csv}")
