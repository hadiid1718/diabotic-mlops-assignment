import pandas as pd
import os

INPUT = os.path.join(os.path.dirname(__file__), '..', 'diabetes_extracted.csv')
OUT = os.path.join(os.path.dirname(__file__), '..', 'diabetes_extracted_fixed.csv')
print('Reading', INPUT)
df = pd.read_csv(INPUT, low_memory=False)
print('Shape:', df.shape)

# Define label tokens
diag_tokens = set(['y','yes','p','positive','1','true','t'])
non_diag_tokens = set(['n','no','0','negative','false','f'])

candidates = []
for c in df.columns:
    try:
        vals = df[c].astype(str).str.strip().str.lower()
    except Exception:
        continue
    counts = vals.value_counts()
    d = {k:v for k,v in counts.items() if str(k) in ['nan'] or str(k)!=''}
    # count diag/non-diag
    diag_count = vals.isin(diag_tokens).sum()
    non_diag_count = vals.isin(non_diag_tokens).sum()
    if diag_count + non_diag_count > 0:
        candidates.append((c, int(diag_count), int(non_diag_count), int(len(vals) - vals.isna().sum())))

# Sort candidates by total label-like counts
candidates = sorted(candidates, key=lambda x: x[1]+x[2], reverse=True)
print('\nCandidate columns with label-like values (col, diag_count, non_diag_count, total_rows):')
for item in candidates[:20]:
    print(item)

# If any candidate has both diag and non_diag counts > 0, pick the best
label_col = None
for c, dc, nc, total in candidates:
    if dc > 0 and nc > 0:
        label_col = c
        print('\nSelected label column:', label_col, 'diag', dc, 'non-diag', nc)
        break

# If not found, construct label from row-wise search across candidates
if label_col is None:
    print('\nNo single column with both classes found. Constructing row-wise label using candidate columns...')
    cand_cols = [c for c,dc,nc,total in candidates if (dc+nc) > 0]
    print('Candidate columns to scan:', cand_cols[:20])
    def row_label(row):
        for c in cand_cols:
            v = row.get(c)
            if pd.isna(v):
                continue
            s = str(v).strip().lower()
            if s in diag_tokens:
                return 'Y'
            if s in non_diag_tokens:
                return 'N'
        return pd.NA
    df['CLASS_fixed'] = df.apply(row_label, axis=1)
else:
    vals = df[label_col].astype(str).str.strip().str.lower()
    def map_label(v):
        if pd.isna(v) or str(v).strip().lower() in ['nan', '']:
            return pd.NA
        s = str(v).strip().lower()
        if s in diag_tokens:
            return 'Y'
        if s in non_diag_tokens:
            return 'N'
        return pd.NA
    df['CLASS_fixed'] = df[label_col].apply(map_label)

print('\nCLASS_fixed value counts:')
print(df['CLASS_fixed'].value_counts(dropna=False))

# Replace existing CLASS column if exists
if 'CLASS' in df.columns:
    df['CLASS'] = df['CLASS_fixed'].where(~df['CLASS_fixed'].isna(), df['CLASS'])
else:
    df['CLASS'] = df['CLASS_fixed']

# Save
print('Saving fixed extracted CSV to', OUT)
df.to_csv(OUT, index=False)
print('Done.')
