import pandas as pd

for f in ['..\\diabetes_extracted.csv','..\\diabetes_cleaned.csv']:
    path = f
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f'Could not read {path}:', e)
        continue
    col = next((c for c in df.columns if c.lower()=='class'), None)
    print('\nFile:', path)
    if col is None:
        print(' No CLASS column')
        continue
    print(' CLASS column name:', col)
    print(df[col].value_counts(dropna=False).head(50))
