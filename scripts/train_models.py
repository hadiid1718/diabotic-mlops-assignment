import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

INPUT = os.path.join(os.path.dirname(__file__), '..', 'diabetes_cleaned.csv')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(OUT_DIR, exist_ok=True)

print('Reading', INPUT)
df = pd.read_csv(INPUT, low_memory=False)
print('Data shape:', df.shape)

# find target column
possible_targets = ('class', 'outcome', 'diabetes', 'target')
class_col = None
for c in df.columns:
    if c.lower() in possible_targets:
        class_col = c
        break

if class_col is None:
    raise SystemExit('No target column found (expected CLASS/outcome/etc)')

print('Using target column:', class_col)

# map target to binary label
def is_diabetic(v):
    if pd.isna(v):
        return pd.NA
    s = str(v).strip().lower()
    diabetic_set = {'y', 'yes', '1', 'positive', 'p', 'true', 't'}
    non_diabetic_set = {'n', 'no', '0', 'negative', 'neg', 'false', 'f'}
    if s in diabetic_set:
        return 1
    if s in non_diabetic_set:
        return 0
    # fallback: unknown -> NA
    return pd.NA

y = df[class_col].apply(is_diabetic)
print('Target value counts (after mapping):')
print(y.value_counts(dropna=False))

# Drop unlabeled rows (where mapping returned NA)
labeled_mask = y.notna()
print('Labeled rows:', int(labeled_mask.sum()), 'out of', len(y))
if labeled_mask.sum() == 0:
    raise SystemExit('No labeled rows found after mapping target. Cannot train.')

y = y[labeled_mask].astype(int)
df = df.loc[labeled_mask].reset_index(drop=True)

# Prepare feature matrix: drop target and non-numeric columns
X = df.drop(columns=[class_col, '__source_page'] if '__source_page' in df.columns else [class_col]).copy()
# select numeric and boolean columns
numeric_cols = X.select_dtypes(include=[np.number, 'bool']).columns.tolist()
X = X[numeric_cols]

print('Feature columns:', len(X.columns))

# ensure no NaNs
X = X.fillna(X.mean())

# train/test split
# if target has only one class, cannot train classifiers
if y.nunique() < 2:
    print('Target variable has only one class after mapping. Cannot train classifiers.')
    print('Target distribution:')
    print(y.value_counts())
    raise SystemExit(1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print('Train/test sizes:', X_train.shape, X_test.shape)

# scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'LogisticRegression': LogisticRegression(solver='liblinear', max_iter=1000),
    'SVM': SVC(kernel='rbf', probability=False),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

results = []
for name, model in models.items():
    print('\nTraining', name)
    # use scaled data for models that benefit
    if name in ('LogisticRegression', 'SVM', 'KNN'):
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    results.append({'model': name, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1})
    print(f"{name} -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

res_df = pd.DataFrame(results).sort_values(by='f1_score', ascending=False)
print('\nModel comparison:')
print(res_df.to_string(index=False))

res_csv = os.path.join(OUT_DIR, 'models_comparison.csv')
res_df.to_csv(res_csv, index=False)
print('Saved comparison to', res_csv)
