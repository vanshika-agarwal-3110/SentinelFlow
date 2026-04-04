import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import datetime

# Initialize environment and configuration
load_dotenv()
DATA_FILE = "crowd_data.csv"

# Ensure the data file exists before attempting to process
if not os.path.exists(DATA_FILE):
    print(f"Error: '{DATA_FILE}' not found. Please run the detection system first.")
    exit()

# Load and format data - Synchronized with main.py 'Timestamp' column
df = pd.read_csv(DATA_FILE)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%H:%M:%S")

def generate_visualizations(df):
    """Generates a suite of charts to visualize crowd behavior."""
    
    # 1. Temporal Trends: Tracking people count per zone over time
    plt.figure(figsize=(10, 5))
    for zone in df["Zone"].unique():
        zone_data = df[df["Zone"] == zone]
        plt.plot(zone_data["Timestamp"], zone_data["Count"], label=zone)
    plt.xlabel("Time of Day")
    plt.ylabel("People Count")
    plt.title("Crowd Flux Over Time")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 2. Density Analysis: Identifying area saturation levels
    plt.figure(figsize=(10, 5))
    for zone in df["Zone"].unique():
        zone_data = df[df["Zone"] == zone]
        plt.plot(zone_data["Timestamp"], zone_data["Density"], label=zone)
    plt.xlabel("Time of Day")
    plt.ylabel("Density Index")
    plt.title("Spatial Density Trends")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 3. Comparative Distribution: Average load across zones
    plt.figure(figsize=(8, 5))
    zone_avgs = df.groupby("Zone")["Count"].mean()
    
    # Distinct color palette for professional visual separation
    colors = ['#5dade2', '#ec7063', '#58d68d']
    zone_avgs.plot(kind='bar', color=colors[:len(zone_avgs)])
    
    plt.xlabel("Zone Location")
    plt.ylabel("Mean People Count")
    plt.title("Average Crowd Distribution by Zone")
    plt.xticks(rotation=0) 
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def compute_zone_stats(df):
    """Calculates key performance metrics for each specific zone."""
    stats = {}
    for zone in df["Zone"].unique():
        zone_df = df[df["Zone"] == zone]
        if zone_df.empty:
            continue

        # Find the exact moment of peak congestion
        peak_row = zone_df.loc[zone_df["Count"].idxmax()]

        stats[zone] = {
            "average_count": float(round(zone_df["Count"].mean(), 2)),
            "max_count": int(zone_df["Count"].max()),
            "peak_time": str(peak_row["Timestamp"].time()),
            "danger_instances": int(len(zone_df[zone_df["Status"] == "DANGER"]))
        }
    return stats

def generate_rule_based_report(df):
    """Internal logic to generate a report if the LLM API is unavailable."""
    report = []

    # Calculate moving average to determine trend
    trend_data = df.groupby("Timestamp")["Count"].mean()
    if not trend_data.empty:
        if trend_data.iloc[-1] > trend_data.iloc[0]:
            report.append("Trend Analysis: Upward trajectory in crowd movement.")
        else:
            report.append("Trend Analysis: Stable or declining crowd levels.")

    # Hotspot detection based on sustained high density
    hotspots = []
    for zone in df["Zone"].unique():
        zone_df = df[df["Zone"] == zone]
        if len(zone_df) == 0: continue

        # Flag zones that spent significant time in DANGER status
        danger_ratio = len(zone_df[zone_df["Status"] == "DANGER"]) / len(zone_df)
        if danger_ratio > 0.5:
            severity = "CRITICAL"
        elif danger_ratio > 0.2:        
            severity = "ELEVATED"
        else:
            severity = "STABLE"
        
        if severity != "STABLE":
            hotspots.append(f"{zone} ({severity})")

    if hotspots:
        report.append(f"Hotspot Alerts: {', '.join(hotspots)}.")
    else:
        report.append("No consistent high-density hotspots detected.")

    # Identify the absolute system peak
    peak_event = df.loc[df["Count"].idxmax()]
    report.append(f"Global Peak: Recorded at {peak_event['Timestamp'].time()} in {peak_event['Zone']} zone.")

    return "\n".join(report)

# OpenAI Integration
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

def generate_llm_report(stats):
    """Uses GPT-4o to transform raw metrics into a narrative summary."""
    if client is None:
        raise Exception("OpenAI API key missing from environment.")

    prompt = f"""
    Acting as a senior safety analyst, summarize these crowd statistics: {stats}
    Identify the highest risk zone, peak intervals, and provide a safety recommendation.
    Format: One analytical paragraph followed by 3 concise bullet points.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a professional security and crowd analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

def safe_generate_llm_report(stats, df):
    """Handles API failures by falling back to rule-based analysis."""
    try:
        return generate_llm_report(stats)
    except Exception as e:
        print(f"Bypassing LLM due to error ({e}). Using local analytics engine.")
        return generate_rule_based_report(df)

if __name__ == "__main__":
    print("\n--- SentinelFlow Analytics Engine ---\n")

    generate_visualizations(df)
    zone_stats = compute_zone_stats(df)
    
    print("ZONE-SPECIFIC METRICS:")
    for zone, data in zone_stats.items():
        print(f" * {zone}: Mean {data['average_count']} | Peak {data['max_count']}")

    print("\n--- EXECUTIVE SUMMARY ---\n")
    final_narrative = safe_generate_llm_report(zone_stats, df)
    print(final_narrative)

    # Exporting the finalized report
    with open("reports.txt", "w") as f:
        f.write("SENTINELFLOW ANALYTICS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("1. EXECUTIVE OVERVIEW\n")
        f.write(f"- Data Points: {len(df)}\n")
        f.write(f"- Mean Crowd Count: {round(df['Count'].mean(), 2)}\n")
        f.write(f"- System-wide Peak: {df['Count'].max()}\n\n")

        f.write("2. SPATIAL PERFORMANCE\n")
        for zone, data in zone_stats.items():
            f.write(f"[{zone}]: Average {data['average_count']}, Max {data['max_count']} (Peak: {data['peak_time']})\n")

        f.write("\n3. STATISTICAL ANOMALIES\n")
        mean_val, std_val = df["Count"].mean(), df["Count"].std()
        anomalies = df[df["Count"] > mean_val + 2 * std_val]
        
        if not anomalies.empty:
            f.write(f"Identified {len(anomalies)} instances of irregular crowd spikes.\n")
        else:
            f.write("No major statistical anomalies detected.\n")

        f.write("\n4. ANALYST NARRATIVE\n")
        f.write(final_narrative)

        f.write("\n\n5. CONCLUSION\n")
        f.write("System analysis confirms successful spatial mapping and risk categorization. "
                "The current data provides a reliable baseline for temporal crowd behavior.")

    print("\nFinal report exported to 'reports.txt'.")