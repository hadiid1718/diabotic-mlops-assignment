Diabetes Prediction — From Data to Deployment

Deliverables (Part 8)
1) data_model.ipynb - Jupyter notebook with EDA, data cleaning and model training
2) app.py - FastAPI application code (already present)
3) diabetes_model.pkl - Saved best model (in models/)
4) training_columns.pkl - Saved training column names (in models/)
5) requirements.txt - Python dependencies (this file)
6) README.md - Project documentation (this file)
7) screenshots/ - Folder with curl command screenshots

Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
# Windows
venv\\Scripts\\activate
# Unix
source venv/bin/activate

pip install -r requirements.txt
```

2. Run the FastAPI app:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

3. Open API docs: http://127.0.0.1:8000/docs

4. Example cURL tests (run in another terminal):

Test 1 - Diabetic-like payload

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"age": 65, "urea": 7.5, "cr": 52.0, "hba1c": 11.2, "chol": 6.1, "tg": 2.8, "hdl": 0.9, "ldl": 3.5, "vldl": 1.2, "bmi": 32.5, "gender": "M"}'
```

Test 2 - Non-diabetic-like payload

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"age": 28, "urea": 4.2, "cr": 48.0, "hba1c": 5.1, "chol": 4.0, "tg": 1.2, "hdl": 1.8, "ldl": 2.1, "vldl": 0.6, "bmi": 22.0, "gender": "F"}'
```

Notes and caveats
- The dataset extraction pipeline originally leaked label tokens into feature columns. Retraining after removing label-derived features is strongly recommended to obtain realistic model performance.
- The `models/` directory contains `diabetes_model.pkl`, `training_columns.pkl`, and `scaler.pkl` produced by the current training script. These are used by `app.py`.

