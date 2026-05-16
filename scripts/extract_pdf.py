import pdfplumber
import pandas as pd
import sys

if len(sys.argv) < 3:
    print("Usage: python extract_pdf.py <input_pdf> <output_csv>")
    sys.exit(1)

input_pdf = sys.argv[1]
output_csv = sys.argv[2]

all_tables = []

with pdfplumber.open(input_pdf) as pdf:
    for i, page in enumerate(pdf.pages, start=1):
        tables = page.extract_tables()
        if not tables:
            continue
        for t in tables:
            if not t:
                continue
            header = t[0]
            rows = t[1:]

            def make_unique(cols):
                seen = {}
                out = []
                for idx, c in enumerate(cols):
                    name = str(c).strip() if c is not None else f"col_{idx}"
                    if name in seen:
                        seen[name] += 1
                        out.append(f"{name}_{seen[name]}")
                    else:
                        seen[name] = 0
                        out.append(name)
                return out

            try:
                uniq_header = make_unique(header)
                df = pd.DataFrame(rows, columns=uniq_header)
            except Exception:
                df = pd.DataFrame(rows)

            df['__source_page'] = i
            all_tables.append(df)

if all_tables:
    out_df = pd.concat([df.reset_index(drop=True) for df in all_tables], ignore_index=True, sort=False)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved {len(all_tables)} table(s) -> {output_csv} (rows={len(out_df)})")
else:
    print("No tables found in the PDF.")
