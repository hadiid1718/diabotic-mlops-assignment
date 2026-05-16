import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

INPUT = os.path.join(os.path.dirname(__file__), '..', 'diabetes_cleaned.csv')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Reading {INPUT}")
df = pd.read_csv(INPUT, low_memory=False)
print(f"Data shape: {df.shape}")

# Helper to determine diabetic label
def is_diabetic_class(v):
    if pd.isna(v):
        return False
    s = str(v).strip().lower()
    return s in ('y', 'yes', '1', 'positive', 'p', 'true')

# 1) Bar chart showing distribution of male and female patients
if 'Gender_F' in df.columns or 'Gender_M' in df.columns:
    f_count = int(df['Gender_F'].sum()) if 'Gender_F' in df.columns else 0
    m_count = int(df['Gender_M'].sum()) if 'Gender_M' in df.columns else 0
    counts = pd.Series({'F': f_count, 'M': m_count})
else:
    if 'Gender' in df.columns:
        counts = df['Gender'].fillna('Unknown').value_counts()
    else:
        counts = pd.Series(dtype=int)

plt.figure(figsize=(6,4))
sns.barplot(x=counts.index, y=counts.values, palette='pastel')
plt.title('Male vs Female Distribution')
plt.ylabel('Count')
plt.xlabel('Gender')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'gender_distribution.png'))
print('Saved gender_distribution.png')
plt.close()

# 2) Histogram of age distribution
if 'AGE' in df.columns:
    plt.figure(figsize=(8,4))
    sns.histplot(df['AGE'].dropna().astype(float), bins=30, kde=False, color='skyblue')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'age_histogram.png'))
    print('Saved age_histogram.png')
    plt.close()

# 3) Histogram of BMI distribution
if 'BMI' in df.columns:
    plt.figure(figsize=(8,4))
    sns.histplot(df['BMI'].dropna().astype(float), bins=30, kde=False, color='salmon')
    plt.title('BMI Distribution')
    plt.xlabel('BMI')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'bmi_histogram.png'))
    print('Saved bmi_histogram.png')
    plt.close()

# Prepare class for coloring
class_col = None
for c in df.columns:
    if c.lower() == 'class' or c.lower() == 'outcome' or c.lower() == 'diabetes':
        class_col = c
        break

if class_col is None:
    df['__class_str'] = 'Unknown'
    class_col = '__class_str'
else:
    df[class_col] = df[class_col].astype(str)

# 4) Scatter plot: BMI vs HbA1c (color-coded by diabetes class)
if 'HbA1c' in df.columns and 'BMI' in df.columns:
    plt.figure(figsize=(7,5))
    sns.scatterplot(data=df, x='BMI', y='HbA1c', hue=class_col, palette='deep', alpha=0.7)
    plt.title('BMI vs HbA1c by Class')
    plt.xlabel('BMI')
    plt.ylabel('HbA1c')
    plt.legend(title=class_col)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'bmi_vs_hba1c_scatter.png'))
    print('Saved bmi_vs_hba1c_scatter.png')
    plt.close()

# 5) Scatter plot: Age vs HbA1c (color-coded by diabetes class)
if 'HbA1c' in df.columns and 'AGE' in df.columns:
    plt.figure(figsize=(7,5))
    sns.scatterplot(data=df, x='AGE', y='HbA1c', hue=class_col, palette='deep', alpha=0.7)
    plt.title('Age vs HbA1c by Class')
    plt.xlabel('Age')
    plt.ylabel('HbA1c')
    plt.legend(title=class_col)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'age_vs_hba1c_scatter.png'))
    print('Saved age_vs_hba1c_scatter.png')
    plt.close()

# 6) Box plot comparing BMI of diabetic vs non-diabetic patients
if 'BMI' in df.columns and class_col is not None:
    # create boolean diabetic flag
    df['_is_diabetic'] = df[class_col].apply(lambda x: is_diabetic_class(x))
    plt.figure(figsize=(6,5))
    sns.boxplot(x='_is_diabetic', y='BMI', data=df)
    plt.title('BMI: Diabetic vs Non-Diabetic')
    plt.xlabel('Is Diabetic')
    plt.ylabel('BMI')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'bmi_boxplot_diabetic_vs_non.png'))
    print('Saved bmi_boxplot_diabetic_vs_non.png')
    plt.close()

print('EDA completed. Figures saved in', OUT_DIR)
