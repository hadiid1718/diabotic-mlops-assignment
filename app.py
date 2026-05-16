from fastapi import FastAPI, HTTPException
from enum import Enum
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title='Diabetes Prediction API')

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'diabetes_model.pkl')
COLS_PATH = os.path.join(MODELS_DIR, 'training_columns.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

if not os.path.exists(MODEL_PATH) or not os.path.exists(COLS_PATH):
    raise RuntimeError('Model artifacts not found. Run training script first.')

model = joblib.load(MODEL_PATH)
training_columns = joblib.load(COLS_PATH)
scaler = None
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)

# Pydantic model
class Gender(str, Enum):
    M = 'M'
    F = 'F'


class PatientData(BaseModel):
    age: float = Field(..., alias='age')
    urea: float
    cr: float
    hba1c: float
    chol: float
    tg: float
    hdl: float
    ldl: float
    vldl: float
    bmi: float
    gender: Gender

def normalize_gender(g):
    if g is None:
        return None
    s = str(g).strip().lower()
    if s in ('f','female','fem'):
        return 'F'
    if s in ('m','male'):
        return 'M'
    return s.upper()

@app.get('/')
def health_check():
    return {'status': 'API is running'}

@app.post('/predict')
def predict(data: PatientData):
    # build empty input matching training columns
    try:
        cols = list(training_columns)
    except Exception:
        raise HTTPException(status_code=500, detail='Training columns malformed')

    X = pd.DataFrame([np.zeros(len(cols))], columns=cols)

    # map numeric inputs to matching column names (case-insensitive)
    mapping = {
        'age': 'age', 'urea': 'urea', 'cr': 'cr', 'hba1c': 'hba1c', 'chol': 'chol',
        'tg': 'tg', 'hdl': 'hdl', 'ldl': 'ldl', 'vldl': 'vldl', 'bmi': 'bmi'
    }
    # for each field, find the training column with same name ignoring case
    for field_name, value in data.dict().items():
        if field_name == 'gender':
            continue
        # find matching column
        matches = [c for c in cols if c.lower() == field_name.lower()]
        if matches:
            X.at[0, matches[0]] = float(value)
        else:
            # try common alternative names (AGE vs age)
            matches = [c for c in cols if c.lower() == field_name.lower()]
            if matches:
                X.at[0, matches[0]] = float(value)
            # else ignore missing feature (leave as 0)

    # handle gender one-hot encoding
    g = normalize_gender(data.gender)
    # find gender one-hot columns (e.g., Gender_F)
    gender_cols = [c for c in cols if c.lower().startswith('gender')]
    if gender_cols:
        # try to match Gender_F or gender_f
        assigned = False
        for gc in gender_cols:
            # extract suffix after underscore
            parts = gc.split('_')
            if len(parts) > 1:
                suffix = parts[1].strip().lower()
                if g and suffix == g.lower():
                    X.at[0, gc] = 1
                    assigned = True
                else:
                    X.at[0, gc] = 0
        if not assigned and len(gender_cols) == 1:
            # single gender column present, set it to 1 (best-effort)
            X.at[0, gender_cols[0]] = 1
    else:
        # if no one-hot gender columns, maybe a single `Gender` numeric column exists
        gender_col = next((c for c in cols if c.lower() == 'gender'), None)
        if gender_col:
            # map F->1, M->0 for example
            X.at[0, gender_col] = 1 if g == 'F' else 0

    # ensure columns order
    X = X[cols]

    # apply scaler if available
    X_proc = X.values
    if scaler is not None:
        try:
            X_proc = scaler.transform(X)
        except Exception:
            # fallback: use raw values
            X_proc = X.values

    # predict
    try:
        pred = model.predict(X_proc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Model prediction failed: {e}')

    label = int(pred[0])
    return {'prediction': int(label), 'label': 'Diabetic' if label == 1 else 'Non-diabetic'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
