import pandas as pd
import os

# Define the folder where data is stored
data_folder = "data"

# List of files to inspect
files = [
    "Plant_1_Generation_Data.csv",
    "Plant_1_Weather_Sensor_Data.csv",
    "Plant_2_Generation_Data.csv",
    "Plant_2_Weather_Sensor_Data.csv"
]

def inspect_data_folder(folder, file_list):
    print(f"🔍 Searching for data in: {os.path.abspath(folder)}\n")
    
    for file_name in file_list:
        # Construct the full path (e.g., data/Plant_1_Generation_Data.csv)
        file_path = os.path.join(folder, file_name)
        
        if os.path.exists(file_path):
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Print a clear, formatted headline
                header = f" FILE: {file_name} "
                print("=" * 70)
                print(header.center(70, "█"))
                print("=" * 70)
                
                # Display first 5 rows
                print(df.head())
                
                # Optional: Display column info and shape
                print(f"\n📈 Shape: {df.shape[0]} rows x {df.shape[1]} columns")
                print("-" * 70 + "\n")
                
            except Exception as e:
                print(f"❌ Error reading {file_name}: {e}\n")
        else:
            print(f"⚠️  Missing: {file_path} (File not found)\n")

if __name__ == "__main__":
    inspect_data_folder(data_folder, files)