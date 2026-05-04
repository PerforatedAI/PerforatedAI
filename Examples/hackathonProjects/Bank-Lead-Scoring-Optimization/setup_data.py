import pandas as pd
import requests
import io

print("Downloading data...")
url = "https://raw.githubusercontent.com/rafiag/DTI2020/main/data/bank.csv"
s = requests.get(url).content
df = pd.read_csv(io.StringIO(s.decode('utf-8')))

if 'duration' in df.columns:
    df = df.drop(columns=['duration'])

# >>> SHUFFLE DATA BEFORE SPLITTING <<<
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)
print("Data Shuffled & Saved.")