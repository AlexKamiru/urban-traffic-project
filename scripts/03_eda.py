"""
03_eda.py
Exploratory Data Analysis for Urban Traffic Prediction Project
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------
# 1. Load merged dataset
# -------------------------

INPUT_PATH = "data/processed/merged_cleaned_data.csv"
OUTPUT_DIR = "outputs/eda_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("📌 Loading dataset...")
df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])

print("Dataset loaded successfully.")
print(df.head())


# --------------------------------------------
# 2. Basic descriptive statistics
# --------------------------------------------

print("\n📌 Generating descriptive statistics...")
desc_path = os.path.join(OUTPUT_DIR, "descriptive_statistics.csv")
df.describe(include="all").to_csv(desc_path)
print(f"Descriptive statistics saved to {desc_path}")


# --------------------------------------------
# 3. Histograms for numerical features
# --------------------------------------------

num_cols = [
    "speed", "volume", "avg_speed", "occupancy",
    "queue_length", "density_level",
    "temperature", "rainfall", "humidity"
]

for col in num_cols:
    if col in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(f"{OUTPUT_DIR}/{col}_hist.png")
        plt.close()
        print(f"Saved histogram: {col}_hist.png")


# --------------------------------------------
# 4. Traffic volume over time
# --------------------------------------------

if "volume" in df.columns:
    print("📊 Plotting traffic volume over time...")
    plt.figure(figsize=(12, 5))
    df.resample("H", on="timestamp")["volume"].mean().plot()
    plt.title("Hourly Traffic Volume Trend")
    plt.xlabel("Time")
    plt.ylabel("Average Volume")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/traffic_volume_over_time.png")
    plt.close()


# --------------------------------------------
# 5. Congestion by intersection
# --------------------------------------------

if "intersection" in df.columns and "density_level" in df.columns:
    print("📌 Plotting congestion levels by intersection...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="intersection", y="density_level")
    plt.title("Congestion Level by Intersection")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/congestion_by_intersection.png")
    plt.close()


# --------------------------------------------
# 6. Vehicle speed distribution per vehicle_id
# --------------------------------------------

if "speed" in df.columns and "vehicle_id" in df.columns:
    print("📈 Plotting speed distribution by vehicle ID...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="vehicle_id", y="speed")
    plt.title("Vehicle Speed Distribution per Vehicle ID")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/speed_by_vehicle_id.png")
    plt.close()


# --------------------------------------------
# 7. Weather impact on traffic volume
# --------------------------------------------

if "rainfall" in df.columns and "volume" in df.columns:
    print("🌧 Analyzing rainfall vs. traffic volume...")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="rainfall", y="volume")
    plt.title("Rainfall vs Traffic Volume")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rainfall_vs_volume.png")
    plt.close()

if "temperature" in df.columns and "speed" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="temperature", y="speed")
    plt.title("Temperature vs Vehicle Speed")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/temperature_vs_speed.png")
    plt.close()


# --------------------------------------------
# 8. Event impact on traffic (boolean column)
# --------------------------------------------

if "event_active" in df.columns:
    print("🎉 Analyzing event impact on traffic...")

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="event_active", y="volume")
    plt.title("Traffic Volume During Events vs Normal Days")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/event_vs_volume.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="event_active", y="speed")
    plt.title("Vehicle Speeds During Events vs Normal Days")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/event_vs_speed.png")
    plt.close()


print("\n✅ EDA COMPLETED SUCCESSFULLY!")
print(f"📂 All plots saved inside: {OUTPUT_DIR}")
