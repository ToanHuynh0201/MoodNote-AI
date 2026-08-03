import pandas as pd

csv_files = ['data/synthetic/qa/audit_sample/rater_a_sheet.csv', "data/synthetic/qa/audit_sample/rater_b_sheet.csv"]

for file in csv_files:
    df = pd.read_csv(file, encoding="utf-8-sig")

    excel_file = file.replace('.csv', '.xlsx')

    df.to_excel(excel_file, index=False)

    print(f"Đã chuyển đổi: {file} -> {excel_file}")