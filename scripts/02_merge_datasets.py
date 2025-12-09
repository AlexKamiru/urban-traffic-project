"""
02_merge_datasets.py
Purpose: Merge GPS, Sensor, Weather, and Events datasets for Nairobi CBD
         urban traffic analysis. Aligns data by timestamp and location.
Author: Traffic Analysis System
Date: 2025
"""

import pandas as pd
import numpy as np
import os
from datetime import timedelta
from math import radians, sin, cos, sqrt, atan2

# ---------------------------
# Define file paths
# ---------------------------
PROCESSED_DATA_DIR = "data/processed"
MERGED_FILE = os.path.join(PROCESSED_DATA_DIR, "merged_cleaned_data.csv")

GPS_FILE = os.path.join(PROCESSED_DATA_DIR, "gps_clean.csv")
SENSOR_FILE = os.path.join(PROCESSED_DATA_DIR, "sensor_clean.csv")
WEATHER_FILE = os.path.join(PROCESSED_DATA_DIR, "weather_clean.csv")
EVENTS_FILE = os.path.join(PROCESSED_DATA_DIR, "events_clean.csv")

# Ensure processed directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# ---------------------------
# Utility functions
# ---------------------------
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate Haversine distance (km) between two GPS points.
    """
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def map_to_nearest_intersection(gps_df):
    """
    Map each GPS point to the nearest intersection using fixed coordinates.
    """
    # Approximate coordinates for major Nairobi CBD intersections
    intersection_coords = {
        "Kenyatta_Ave": {"lat": -1.286389, "lon": 36.821111},
        "Haile_Selassie": {"lat": -1.283333, "lon": 36.816667},
        "Globe": {"lat": -1.292500, "lon": 36.816944},
        "Moi_Avenue": {"lat": -1.288056, "lon": 36.823611},
        "Uhuru_Highway": {"lat": -1.292222, "lon": 36.819444},
        "CBD": {"lat": -1.286389, "lon": 36.821111},
        "KICC": {"lat": -1.292065, "lon": 36.821946},
        "Uhuru_Park": {"lat": -1.292500, "lon": 36.817500},
        "Kasarani": {"lat": -1.219722, "lon": 36.927222}
    }

    nearest_intersections = []
    for idx, row in gps_df.iterrows():
        min_dist = float("inf")
        nearest = None
        for inter, coords in intersection_coords.items():
            dist = haversine(row['latitude'], row['longitude'], coords['lat'], coords['lon'])
            if dist < min_dist:
                min_dist = dist
                nearest = inter
        nearest_intersections.append(nearest)

    gps_df['intersection'] = nearest_intersections
    return gps_df

# ---------------------------
# Load cleaned datasets
# ---------------------------
print("\n Loading cleaned datasets...")
gps_df = pd.read_csv(GPS_FILE, parse_dates=['timestamp'])
sensor_df = pd.read_csv(SENSOR_FILE, parse_dates=['timestamp'])
weather_df = pd.read_csv(WEATHER_FILE, parse_dates=['timestamp'])
events_df = pd.read_csv(EVENTS_FILE, parse_dates=['start_time', 'end_time'])

# ---------------------------
# Map GPS points to intersections
# ---------------------------
print("\n Mapping GPS points to nearest intersections...")
gps_df = map_to_nearest_intersection(gps_df)

# ---------------------------
# Merge Sensor data
# ---------------------------
print("\n Merging GPS and Sensor data...")
merged_df = pd.merge(
    gps_df,
    sensor_df,
    on=['intersection', 'timestamp'],
    how='left',
    suffixes=('_gps', '_sensor')
)

# ---------------------------
# Merge Weather data (nearest timestamp)
# ---------------------------
print("\n Merging Weather data...")
# Sort by timestamp for merge_asof
merged_df = merged_df.sort_values('timestamp')
weather_df = weather_df.sort_values('timestamp')

# Use merge_asof to merge on nearest previous timestamp
merged_df = pd.merge_asof(
    merged_df,
    weather_df,
    on='timestamp',
    direction='nearest',  # can also try 'backward' or 'forward'
    suffixes=('', '_weather')
)


# ---------------------------
# Merge Events data (overlapping time)
# ---------------------------
print("\n Merging Events data...")
def is_event_active(row, events_df):
    """
    Check if an event is active at the GPS point's timestamp and intersection.
    """
    active_events = events_df[
        (events_df['start_time'] <= row['timestamp']) &
        (events_df['end_time'] >= row['timestamp']) &
        (events_df['location'].str.lower() == row['intersection'].lower())
    ]
    return not active_events.empty

merged_df['event_active'] = merged_df.apply(lambda row: is_event_active(row, events_df), axis=1)

# ---------------------------
# Save merged dataset
# ---------------------------
merged_df.to_csv(MERGED_FILE, index=False)
print(f"\n✓ Merged dataset saved to: {MERGED_FILE}")
print(f"  - Rows: {len(merged_df):,}")
print(f"  - Columns: {len(merged_df.columns)}")

# ---------------------------
# Preview merged data
# ---------------------------
print("\n--- MERGED DATA SAMPLE ---")
print(merged_df.head())
print(merged_df.info())

print("\n Data merging complete!")
