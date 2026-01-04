import pandas as pd

# Load the timestamped CSV
csv_file = 'PAI/PAI_04.01.2026.16.23.47_noImprove_lr_0Scores.csv'
df = pd.read_csv(csv_file)

print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"\nEpoch range: {df['Epochs'].min()} to {df['Epochs'].max()}")

# Check for data completeness
df_clean = df.dropna(subset=['Train Loss'])
print(f"\nRows with Train Loss data: {len(df_clean)}")
print(f"Epochs with data: {sorted(df_clean['Epochs'].unique())}")

print("\nFirst 10 rows:")
print(df_clean.head(10))

print("\nLast 10 rows:")
print(df_clean.tail(10))
