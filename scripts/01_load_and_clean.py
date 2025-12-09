"""
01_load_and_clean.py
Purpose: Load, clean, and preprocess GPS, Sensor, Weather, and Events datasets
         for Nairobi CBD urban traffic analysis.
"""

import pandas as pd
import numpy as np
import os

# ---------------------------
# Define file paths
# ---------------------------
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

GPS_FILE = os.path.join(RAW_DATA_DIR, "gps_data.csv")
SENSOR_FILE = os.path.join(RAW_DATA_DIR, "sensor_data.csv")
WEATHER_FILE = os.path.join(RAW_DATA_DIR, "weather_data.csv")
EVENTS_FILE = os.path.join(RAW_DATA_DIR, "events_data.csv")

# Ensure processed directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# ---------------------------
# Utility functions
# ---------------------------
def standardize_columns(df):
    """Convert column names to lowercase snake_case"""
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df

def parse_datetime(df, column_name, fmt=None):
    """Convert a column to pandas datetime"""
    df[column_name] = pd.to_datetime(df[column_name], format=fmt, errors='coerce')
    return df

def handle_missing(df, strategy="drop", fill_value=None):
    """Handle missing values: drop or fill"""
    if strategy == "drop":
        return df.dropna()
    elif strategy == "fill":
        return df.fillna(fill_value)
    else:
        return df

# ---------------------------
# Load datasets
# ---------------------------
print("\n Loading datasets...")

gps_df = pd.read_csv(GPS_FILE)
sensor_df = pd.read_csv(SENSOR_FILE)
weather_df = pd.read_csv(WEATHER_FILE)
events_df = pd.read_csv(EVENTS_FILE)

# ---------------------------
# Standardize column names
# ---------------------------
gps_df = standardize_columns(gps_df)
sensor_df = standardize_columns(sensor_df)
weather_df = standardize_columns(weather_df)
events_df = standardize_columns(events_df)

# ---------------------------
# Parse timestamps
# ---------------------------
gps_df = parse_datetime(gps_df, "timestamp")
sensor_df = parse_datetime(sensor_df, "timestamp")
weather_df = parse_datetime(weather_df, "timestamp")
events_df = parse_datetime(events_df, "start_time")
events_df = parse_datetime(events_df, "end_time")

# ---------------------------
# Handle duplicates
# ---------------------------
gps_df = gps_df.drop_duplicates()
sensor_df = sensor_df.drop_duplicates()
weather_df = weather_df.drop_duplicates()
events_df = events_df.drop_duplicates()

# ---------------------------
# Handle missing values
# ---------------------------
# GPS: drop rows with missing coordinates or speed
gps_df = gps_df.dropna(subset=["latitude", "longitude", "speed"])

# Sensor: fill missing numeric values with 0
sensor_df = sensor_df.fillna({
    "volume": 0,
    "avgspeed": 0,
    "occupancy": 0,
    "queue_length": 0,
    "density_level": 0
})

# Weather: forward-fill missing values
weather_df = weather_df.fillna(method="ffill")

# Events: fill missing event types with "unknown"
events_df["type"] = events_df["type"].fillna("unknown")
events_df["expected_attendance"] = events_df["expected_attendance"].fillna(0)

# ---------------------------
# Normalize GPS & Sensor speed
# ---------------------------
# GPS speed: clip between 0-80 km/h
gps_df["speed"] = gps_df["speed"].clip(lower=0, upper=80)

# Sensor average speed: clip reasonable bounds (0-120 km/h)
sensor_df["avg_Speed"] = sensor_df["avg_speed"].clip(lower=0, upper=120)

# ---------------------------
# Encode categorical features
# ---------------------------
# Events type: one-hot encoding for modeling
events_encoded = pd.get_dummies(events_df, columns=["type"], prefix="event")

# ---------------------------
# Save cleaned datasets
# ---------------------------
gps_clean_file = os.path.join(PROCESSED_DATA_DIR, "gps_clean.csv")
sensor_clean_file = os.path.join(PROCESSED_DATA_DIR, "sensor_clean.csv")
weather_clean_file = os.path.join(PROCESSED_DATA_DIR, "weather_clean.csv")
events_clean_file = os.path.join(PROCESSED_DATA_DIR, "events_clean.csv")

gps_df.to_csv(gps_clean_file, index=False)
sensor_df.to_csv(sensor_clean_file, index=False)
weather_df.to_csv(weather_clean_file, index=False)
events_encoded.to_csv(events_clean_file, index=False)

print("\n Cleaned datasets saved to 'data/processed/'")

# ---------------------------
# Preview datasets
# ---------------------------
def preview(df, name, n=5):
    print(f"\n--- {name.upper()} ---")
    print(df.head(n))
    print(df.info())

preview(gps_df, "GPS Data")
preview(sensor_df, "Sensor Data")
preview(weather_df, "Weather Data")
preview(events_encoded, "Events Data (Encoded)")

print("\n Data loading and cleaning complete!")
