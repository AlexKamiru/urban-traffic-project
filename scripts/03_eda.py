"""
03_eda.py - Enhanced Version
Exploratory Data Analysis for Urban Traffic Prediction Project
With improved visualizations for better readability and interpretation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# Set style for better visual appeal
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# -------------------------
# 1. Load merged dataset
# -------------------------

INPUT_PATH = "data/processed/merged_cleaned_data.csv"
OUTPUT_DIR = "outputs/eda_plots_enhanced"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])

print("Dataset loaded successfully.")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nData types:")
print(df.dtypes)
print("\nFirst few rows:")
print(df.head())

# Handle duplicate column names
if 'avg_speed' in df.columns and 'avg_Speed' in df.columns:
    # Keep avg_speed and drop avg_Speed if they're the same
    if df['avg_speed'].equals(df['avg_Speed']):
        df = df.drop(columns=['avg_Speed'])
        print("\nDropped duplicate column 'avg_Speed'")

# --------------------------------------------
# 2. Basic descriptive statistics
# --------------------------------------------

print("\nGenerating descriptive statistics...")
desc_path = os.path.join(OUTPUT_DIR, "descriptive_statistics.csv")
desc_stats = df.describe(include="all")
desc_stats.to_csv(desc_path)
print(f"Descriptive statistics saved to {desc_path}")

# Additional summary
print("\nMissing values per column:")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Check categorical columns
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns: {cat_cols}")

# --------------------------------------------
# 3. Enhanced histograms for numerical features
# --------------------------------------------

# Select only numeric columns for histograms
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove any non-traffic numeric columns if needed
exclude_from_hist = ['latitude', 'longitude']  # Add any other non-numeric looking columns
num_cols = [col for col in num_cols if col not in exclude_from_hist]

print(f"\nCreating enhanced histograms for {len(num_cols)} numerical columns...")

for col in num_cols[:15]:  # Limit to first 15 to avoid too many plots
    plt.figure(figsize=(10, 6))
    
    # Create subplot layout
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram with KDE and statistics
    sns.histplot(data=df, x=col, kde=True, ax=axes[0], bins=30, 
                 edgecolor='black', linewidth=0.5)
    axes[0].axvline(df[col].mean(), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {df[col].mean():.2f}')
    axes[0].axvline(df[col].median(), color='green', linestyle='--', 
                    linewidth=2, label=f'Median: {df[col].median():.2f}')
    axes[0].set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel(col, fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    sns.boxplot(data=df, y=col, ax=axes[1], color='skyblue')
    axes[1].set_title(f'Box Plot of {col}', fontsize=14, fontweight='bold')
    axes[1].set_ylabel(col, fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Add statistics as text
    stats_text = f"""
    Statistics:
    • Mean: {df[col].mean():.2f}
    • Median: {df[col].median():.2f}
    • Std Dev: {df[col].std():.2f}
    • Min: {df[col].min():.2f}
    • Max: {df[col].max():.2f}
    • Skewness: {df[col].skew():.2f}
    """
    axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{col}_enhanced_dist.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved enhanced distribution plot: {col}_enhanced_dist.png")

# --------------------------------------------
# 4. Enhanced traffic volume over time
# --------------------------------------------

if "volume" in df.columns:
    print("\nPlotting enhanced traffic volume over time...")
    
    # Create time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['date'] = df['timestamp'].dt.date
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Hourly trend
    hourly_avg = df.groupby('hour')['volume'].mean()
    axes[0, 0].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2)
    axes[0, 0].fill_between(hourly_avg.index, hourly_avg.values, alpha=0.3)
    axes[0, 0].set_title('Average Hourly Traffic Volume', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Hour of Day', fontsize=12)
    axes[0, 0].set_ylabel('Average Volume', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvspan(7, 9, alpha=0.2, color='red', label='Morning Peak')
    axes[0, 0].axvspan(16, 18, alpha=0.2, color='orange', label='Evening Peak')
    axes[0, 0].legend()
    
    # 2. Daily trend
    daily_avg = df.groupby('day_of_week')['volume'].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[0, 1].bar(days, daily_avg.values, color='steelblue', alpha=0.7)
    axes[0, 1].set_title('Average Daily Traffic Volume', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Day of Week', fontsize=12)
    axes[0, 1].set_ylabel('Average Volume', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Monthly trend
    monthly_avg = df.groupby('month')['volume'].mean()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_labels = [months[i-1] for i in monthly_avg.index]
    axes[1, 0].plot(monthly_labels, monthly_avg.values, marker='s', linewidth=2, color='green')
    axes[1, 0].set_title('Monthly Traffic Volume Trend', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Month', fontsize=12)
    axes[1, 0].set_ylabel('Average Volume', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Time series with rolling average
    daily_volume = df.groupby('date')['volume'].mean()
    axes[1, 1].plot(daily_volume.index, daily_volume.values, alpha=0.7, label='Daily Average')
    rolling_avg = daily_volume.rolling(window=7, min_periods=1).mean()
    axes[1, 1].plot(daily_volume.index, rolling_avg, 
                   linewidth=2, color='red', label='7-day Rolling Avg')
    axes[1, 1].set_title('Traffic Volume Time Series', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Date', fontsize=12)
    axes[1, 1].set_ylabel('Volume', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/traffic_volume_temporal_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved comprehensive temporal analysis")

# --------------------------------------------
# 5. Enhanced congestion by intersection
# --------------------------------------------

if "intersection" in df.columns:
    print("\nPlotting enhanced congestion levels by intersection...")
    
    # Get top N intersections for readability
    top_n = min(10, df['intersection'].nunique())
    top_intersections = df['intersection'].value_counts().head(top_n).index
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Volume by intersection
    intersection_volume = df[df['intersection'].isin(top_intersections)].groupby('intersection')['volume'].mean().sort_values()
    axes[0].barh(range(len(intersection_volume)), intersection_volume.values, color='steelblue')
    axes[0].set_yticks(range(len(intersection_volume)))
    axes[0].set_yticklabels(intersection_volume.index)
    axes[0].set_title(f'Average Volume by Intersection\n(Top {top_n})', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Average Volume', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # 2. Handle density_level - check if it's categorical or numeric
    if "density_level" in df.columns:
        # Check data type
        if df['density_level'].dtype == 'object' or df['density_level'].dtype.name == 'category':
            # Categorical: create stacked bar chart
            df_filtered = df[df['intersection'].isin(top_intersections)].copy()
            
            # Clean density_level values
            df_filtered['density_level_clean'] = df_filtered['density_level'].astype(str).str.lower().str.strip()
            
            # Count occurrences
            density_counts = pd.crosstab(df_filtered['intersection'], 
                                        df_filtered['density_level_clean'],
                                        normalize='index')
            
            # Sort by total count
            density_counts = density_counts.loc[intersection_volume.index]
            
            # Plot
            density_counts.plot(kind='bar', stacked=True, ax=axes[1], 
                              colormap='RdYlGn_r', edgecolor='black', linewidth=0.5)
            axes[1].set_title(f'Density Level Distribution by Intersection\n(Top {top_n})', 
                             fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Intersection', fontsize=12)
            axes[1].set_ylabel('Percentage', fontsize=12)
            axes[1].legend(title='Density Level', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3, axis='y')
        else:
            # Numeric: create box plot
            df_filtered = df[df['intersection'].isin(top_intersections)]
            sns.boxplot(data=df_filtered, x='intersection', y='density_level', ax=axes[1])
            axes[1].set_title(f'Density Level by Intersection\n(Top {top_n})', 
                             fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Intersection', fontsize=12)
            axes[1].set_ylabel('Density Level', fontsize=12)
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3, axis='y')
    
    # 3. Heatmap of hourly patterns (using volume since it's numeric)
    df_filtered = df[df['intersection'].isin(top_intersections)].copy()
    
    # Create pivot table with volume (which is numeric)
    try:
        pivot_table = df_filtered.pivot_table(
            values='volume',
            index='intersection',
            columns='hour',
            aggfunc='mean'
        )
        
        # Sort by intersection order from first plot
        pivot_table = pivot_table.reindex(intersection_volume.index)
        
        sns.heatmap(pivot_table, cmap='YlOrRd', ax=axes[2], 
                   cbar_kws={'label': 'Average Volume'})
        axes[2].set_title(f'Hourly Volume Pattern by Intersection\n(Top {top_n})', 
                         fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Hour of Day', fontsize=12)
        axes[2].set_ylabel('Intersection', fontsize=12)
    except Exception as e:
        # Fallback: event distribution by intersection
        if 'event_active' in df.columns:
            event_rate = df_filtered.groupby('intersection')['event_active'].mean()
            event_rate = event_rate.reindex(intersection_volume.index)
            axes[2].bar(range(len(event_rate)), event_rate.values, color='orange')
            axes[2].set_xticks(range(len(event_rate)))
            axes[2].set_xticklabels(event_rate.index, rotation=45)
            axes[2].set_title(f'Event Rate by Intersection\n(Top {top_n})', 
                             fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Intersection', fontsize=12)
            axes[2].set_ylabel('Event Rate', fontsize=12)
            axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/intersection_analysis_enhanced.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved enhanced intersection analysis")

# --------------------------------------------
# 6. Enhanced vehicle speed analysis
# --------------------------------------------

if "speed" in df.columns:
    print("\nAnalyzing vehicle speed patterns...")
    
    # Create subplots based on available data
    available_plots = []
    if 'hour' in df.columns:
        available_plots.append('hourly')
    if 'volume' in df.columns:
        available_plots.append('volume')
    if 'density_level' in df.columns and df['density_level'].dtype != 'object':
        available_plots.append('density')
    
    n_plots = len(available_plots)
    if n_plots > 0:
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        if n_plots == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # 1. Speed distribution by time of day
        if 'hourly' in available_plots:
            hourly_speed = df.groupby('hour')['speed'].agg(['mean', 'std']).reset_index()
            axes[plot_idx].errorbar(hourly_speed['hour'], hourly_speed['mean'], 
                                   yerr=hourly_speed['std'], fmt='o-', capsize=5, linewidth=2)
            axes[plot_idx].set_title('Average Speed by Hour of Day', fontsize=14, fontweight='bold')
            axes[plot_idx].set_xlabel('Hour', fontsize=12)
            axes[plot_idx].set_ylabel('Average Speed', fontsize=12)
            axes[plot_idx].grid(True, alpha=0.3)
            axes[plot_idx].axvspan(7, 9, alpha=0.2, color='red', label='Rush Hour')
            axes[plot_idx].legend()
            plot_idx += 1
        
        # 2. Speed vs Volume scatter with regression
        if 'volume' in available_plots:
            # Sample for performance if large dataset
            sample_size = min(5000, len(df))
            sample_df = df.sample(sample_size) if len(df) > sample_size else df
            
            sns.scatterplot(data=sample_df, x='volume', y='speed', alpha=0.5, ax=axes[plot_idx], s=10)
            
            # Add regression line
            try:
                z = np.polyfit(sample_df['volume'], sample_df['speed'], 1)
                p = np.poly1d(z)
                axes[plot_idx].plot(sample_df['volume'], p(sample_df['volume']), "r--", linewidth=2)
            except:
                pass
            
            # Calculate correlation
            try:
                corr = sample_df[['volume', 'speed']].corr().iloc[0,1]
                axes[plot_idx].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[plot_idx].transAxes,
                                   fontsize=10, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except:
                pass
            
            axes[plot_idx].set_title('Speed vs Traffic Volume', fontsize=14, fontweight='bold')
            axes[plot_idx].set_xlabel('Volume', fontsize=12)
            axes[plot_idx].set_ylabel('Speed', fontsize=12)
            axes[plot_idx].grid(True, alpha=0.3)
            plot_idx += 1
        
        # 3. Speed distribution by density level
        if 'density' in available_plots:
            # Convert density_level to categorical bins if numeric
            if df['density_level'].dtype != 'object':
                # Create quantile-based bins
                df['density_bin'] = pd.qcut(df['density_level'], q=3, labels=['Low', 'Medium', 'High'])
                density_col = 'density_bin'
            else:
                density_col = 'density_level'
            
            sns.boxplot(data=df, x=density_col, y='speed', ax=axes[plot_idx])
            axes[plot_idx].set_title('Speed Distribution by Traffic Density', 
                                    fontsize=14, fontweight='bold')
            axes[plot_idx].set_xlabel('Density Level', fontsize=12)
            axes[plot_idx].set_ylabel('Speed', fontsize=12)
            axes[plot_idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/speed_analysis_enhanced.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved enhanced speed analysis")

# --------------------------------------------
# 7. Enhanced weather impact analysis
# --------------------------------------------

print("\nAnalyzing enhanced weather impact on traffic...")

# Rainfall vs Volume - with bins and trend
if "rainfall" in df.columns and "volume" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter with regression and confidence interval
    sample_size = min(5000, len(df))
    sample_df = df.sample(sample_size) if len(df) > sample_size else df
    
    # Remove rows with missing volume
    sample_df = sample_df.dropna(subset=['volume', 'rainfall'])
    
    try:
        sns.regplot(data=sample_df, x="rainfall", y="volume", 
                   scatter_kws={'alpha': 0.3, 's': 10}, 
                   line_kws={'color': 'red', 'linewidth': 2},
                   ci=95, ax=axes[0])
        axes[0].set_title("Rainfall vs Traffic Volume\n(with 95% CI)", 
                         fontsize=14, fontweight='bold')
    except:
        sns.scatterplot(data=sample_df, x="rainfall", y="volume", 
                       alpha=0.3, s=10, ax=axes[0])
        axes[0].set_title("Rainfall vs Traffic Volume", 
                         fontsize=14, fontweight='bold')
    
    axes[0].set_xlabel("Rainfall", fontsize=12)
    axes[0].set_ylabel("Volume", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Binned analysis
    try:
        rainfall_bins = [0, 0.1, 2, 5, 10, df['rainfall'].max()]
        bin_labels = ['None', 'Light', 'Moderate', 'Heavy', 'Very Heavy']
        
        # Ensure we have valid bins
        if len(set(rainfall_bins)) >= 3:
            df['rainfall_bin'] = pd.cut(df['rainfall'], bins=rainfall_bins, labels=bin_labels[:len(rainfall_bins)-1])
            rain_volume = df.groupby('rainfall_bin')['volume'].agg(['mean', 'std', 'count']).dropna()
            
            if len(rain_volume) > 0:
                x_pos = np.arange(len(rain_volume))
                axes[1].bar(x_pos, rain_volume['mean'], yerr=rain_volume['std'], 
                           capsize=5, alpha=0.7, color='steelblue')
                axes[1].set_xticks(x_pos)
                axes[1].set_xticklabels(rain_volume.index, rotation=45)
                axes[1].set_title("Average Volume by Rainfall Intensity", 
                                 fontsize=14, fontweight='bold')
                axes[1].set_xlabel("Rainfall Intensity", fontsize=12)
                axes[1].set_ylabel("Average Volume", fontsize=12)
                axes[1].grid(True, alpha=0.3, axis='y')
                
                # Add count annotations
                for i, count in enumerate(rain_volume['count']):
                    if i < len(rain_volume['mean']):
                        axes[1].text(i, rain_volume['mean'].iloc[i] + rain_volume['std'].iloc[i], 
                                    f'n={int(count)}', ha='center', va='bottom', fontsize=8)
            else:
                axes[1].text(0.5, 0.5, 'Insufficient data\nfor binned analysis', 
                            ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
                axes[1].set_title("Rainfall Binned Analysis", fontsize=14, fontweight='bold')
        else:
            axes[1].text(0.5, 0.5, 'Insufficient rainfall variation\nfor binned analysis', 
                        ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
            axes[1].set_title("Rainfall Binned Analysis", fontsize=14, fontweight='bold')
    except Exception as e:
        axes[1].text(0.5, 0.5, f'Error in binned analysis:\n{str(e)[:50]}...', 
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=10)
        axes[1].set_title("Rainfall Binned Analysis", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rainfall_impact_enhanced.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved enhanced rainfall analysis")

# Temperature vs Speed - with non-linear analysis
if "temperature" in df.columns and "speed" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter with LOESS/regression
    sample_size = min(5000, len(df))
    sample_df = df.sample(sample_size) if len(df) > sample_size else df
    sample_df = sample_df.dropna(subset=['temperature', 'speed'])
    
    # Scatter plot
    sns.scatterplot(data=sample_df, x="temperature", y="speed", 
                   alpha=0.4, s=15, ax=axes[0])
    
    # Try polynomial fit
    try:
        coeffs = np.polyfit(sample_df['temperature'], sample_df['speed'], 2)
        poly = np.poly1d(coeffs)
        x_sorted = np.sort(sample_df['temperature'])
        axes[0].plot(x_sorted, poly(x_sorted), color='red', linewidth=2, 
                    label=f'Quadratic fit')
    except:
        # Fallback to linear
        try:
            sns.regplot(data=sample_df, x="temperature", y="speed", 
                       scatter=False, ci=None, ax=axes[0], color='red', label='Linear fit')
        except:
            pass
    
    axes[0].set_title("Temperature vs Vehicle Speed", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Temperature", fontsize=12)
    axes[0].set_ylabel("Speed", fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Binned box plot
    try:
        # Create temperature bins
        temp_bins = pd.cut(df['temperature'], bins=5)
        df_temp = df.copy()
        df_temp['temp_bin'] = temp_bins
        
        # Remove rows where bin is NaN
        df_temp = df_temp.dropna(subset=['temp_bin', 'speed'])
        
        if len(df_temp['temp_bin'].unique()) > 1:
            sns.boxplot(data=df_temp, x='temp_bin', y='speed', ax=axes[1])
            axes[1].set_title("Speed Distribution by Temperature Range", 
                             fontsize=14, fontweight='bold')
            axes[1].set_xlabel("Temperature Range", fontsize=12)
            axes[1].set_ylabel("Speed", fontsize=12)
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3, axis='y')
        else:
            axes[1].text(0.5, 0.5, 'Insufficient temperature variation', 
                        ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
            axes[1].set_title("Temperature Binned Analysis", fontsize=14, fontweight='bold')
    except:
        axes[1].text(0.5, 0.5, 'Error in binned analysis', 
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
        axes[1].set_title("Temperature Binned Analysis", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/temperature_impact_enhanced.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved enhanced temperature analysis")

# --------------------------------------------
# 8. Enhanced event impact analysis
# --------------------------------------------

if "event_active" in df.columns:
    print("\nAnalyzing enhanced event impact on traffic...")
    
    # Create subplots based on available metrics
    event_metrics = []
    if 'volume' in df.columns:
        event_metrics.append('volume')
    if 'speed' in df.columns:
        event_metrics.append('speed')
    if 'occupancy' in df.columns:
        event_metrics.append('occupancy')
    if 'queue_length' in df.columns:
        event_metrics.append('queue_length')
    
    n_metrics = len(event_metrics)
    if n_metrics > 0:
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        
        # Flatten axes array for easier indexing
        if n_metrics > 1:
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]
        
        for idx, metric in enumerate(event_metrics):
            ax = axes_flat[idx]
            
            # Box plot
            sns.boxplot(data=df, x="event_active", y=metric, ax=ax)
            
            # Add percentage change if possible
            try:
                normal_val = df[df['event_active'] == False][metric].mean()
                event_val = df[df['event_active'] == True][metric].mean()
                
                if pd.notna(normal_val) and pd.notna(event_val) and normal_val != 0:
                    pct_change = ((event_val - normal_val) / normal_val) * 100
                    ax.text(0.5, 0.95, f'Change: {pct_change:+.1f}%', 
                           transform=ax.transAxes, ha='center',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            except:
                pass
            
            # Title and labels
            titles = {
                'volume': 'Traffic Volume',
                'speed': 'Vehicle Speed', 
                'occupancy': 'Road Occupancy',
                'queue_length': 'Queue Length'
            }
            ax.set_title(f"{titles.get(metric, metric)}: Events vs Normal Days", 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel("Event Active", fontsize=12)
            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide unused subplots
        for idx in range(len(event_metrics), len(axes_flat)):
            axes_flat[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/event_impact_enhanced.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved enhanced event impact analysis")

# --------------------------------------------
# 9. Correlation matrix
# --------------------------------------------

print("\nCreating correlation analysis...")

# Select numerical columns for correlation
corr_cols = [col for col in num_cols if col in df.columns and df[col].dtype != 'object']
if len(corr_cols) > 1:
    # Calculate correlation
    corr_data = df[corr_cols].dropna()
    
    if len(corr_data) > 0 and len(corr_cols) > 1:
        corr_matrix = corr_data.corr()
        
        plt.figure(figsize=(12, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                   center=0, mask=mask, square=True, linewidths=0.5,
                   cbar_kws={"shrink": 0.8})
        plt.title("Correlation Matrix of Numerical Features", 
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/correlation_matrix.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved correlation matrix")
        
        # Print top correlations
        print("\nTop positive correlations:")
        corr_pairs = corr_matrix.unstack()
        sorted_pairs = corr_pairs.sort_values(kind="quicksort", ascending=False)
        for idx, (pair, value) in enumerate(sorted_pairs.items()):
            if pair[0] != pair[1] and value > 0.5:
                print(f"  {pair[0]} - {pair[1]}: {value:.3f}")
    else:
        print("  Insufficient data for correlation matrix")
else:
    print("  Not enough numerical columns for correlation analysis")

# --------------------------------------------
# 10. Summary report
# --------------------------------------------

print("\nGenerating summary report...")
with open(f"{OUTPUT_DIR}/eda_summary.txt", "w") as f:
    f.write("="*60 + "\n")
    f.write("EXPLORATORY DATA ANALYSIS SUMMARY\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"Dataset Information:\n")
    f.write(f"- Shape: {df.shape}\n")
    f.write(f"- Time range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
    f.write(f"- Total columns: {len(df.columns)}\n")
    f.write(f"- Numerical columns: {len(num_cols)}\n")
    f.write(f"- Categorical columns: {len(cat_cols)}\n\n")
    
    f.write(f"Missing Values Summary:\n")
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    if len(missing_cols) > 0:
        for col, count in missing_cols.items():
            f.write(f"- {col}: {count} missing ({count/len(df)*100:.1f}%)\n")
    else:
        f.write("- No missing values found\n")
    
    f.write(f"\nKey Statistics:\n")
    key_metrics = ['volume', 'speed', 'occupancy', 'queue_length', 'temperature', 'rainfall']
    for col in key_metrics:
        if col in df.columns and df[col].dtype != 'object':
            f.write(f"- {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, range=[{df[col].min():.2f}, {df[col].max():.2f}]\n")
    
    f.write("\nTraffic Pattern Insights:\n")
    if 'volume' in df.columns and 'hour' in df.columns:
        peak_hour = df.groupby('hour')['volume'].mean().idxmax()
        f.write(f"- Peak traffic hour: {peak_hour}:00\n")
    
    if 'event_active' in df.columns:
        event_count = df['event_active'].sum()
        f.write(f"- Events occurred in {event_count} records ({event_count/len(df)*100:.1f}% of data)\n")
    
    f.write("\nWeather Impact Insights:\n")
    if 'rainfall' in df.columns:
        rainy_days = (df['rainfall'] > 0).sum()
        f.write(f"- Rainy records: {rainy_days} ({rainy_days/len(df)*100:.1f}%)\n")
    
    f.write("\nPlots Generated:\n")
    plot_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
    for filename in sorted(plot_files):
        f.write(f"- {filename}\n")
    
    f.write(f"\nTotal plots generated: {len(plot_files)}\n")

print("\n" + "="*60)
print("ENHANCED EDA COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\nAll enhanced plots saved to: {OUTPUT_DIR}")
print(f"Summary report: {OUTPUT_DIR}/eda_summary.txt")
print(f"Total plots generated: {len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])}")

# Print key findings
print("\n" + "="*60)
print("KEY FINDINGS SUMMARY")
print("="*60)

if 'volume' in df.columns:
    print(f"\nTraffic Volume:")
    print(f"  • Average: {df['volume'].mean():.1f}")
    print(f"  • Range: {df['volume'].min():.1f} - {df['volume'].max():.1f}")

if 'speed' in df.columns:
    print(f"\nVehicle Speed:")
    print(f"  • Average: {df['speed'].mean():.1f} km/h")
    print(f"  • Typical urban range: {df['speed'].quantile(0.25):.1f} - {df['speed'].quantile(0.75):.1f} km/h")

if 'event_active' in df.columns:
    event_rate = df['event_active'].mean() * 100
    print(f"\nEvents:")
    print(f"  • Event occurrence rate: {event_rate:.1f}%")

if 'rainfall' in df.columns:
    dry_percentage = (df['rainfall'] == 0).mean() * 100
    print(f"\nWeather:")
    print(f"  • Dry conditions: {dry_percentage:.1f}% of time")