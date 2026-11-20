"""
WiFiScope - Wi-Fi Heatmap Floor Plan
------------------------------------
Uses NumPy and Pandas for data cleaning and analysis.

How to run:   python wifiscope.py
Keep the original wifiscope.csv file in the data folder.
Results are saved in the outputs folder.
"""

import numpy as np
import pandas as pd

from src.load_data import load_data
from src.clean_data import clean_data

# =========================================================
# STEP 1: LOAD, CLEAN, AND SAVE THE DATA
# =========================================================
raw_df = load_data("data\\wifiscope.csv")
print("Loaded original data:", raw_df.shape)

# Clean the original data and save the cleaned version for reuse.
cleaned_df, cleaning_report = clean_data(raw_df)
cleaned_df.to_csv("outputs\\measurements_clean.csv", index=False)

with open("outputs\\cleaning_report.txt", "w", encoding="utf-8") as file:
    file.write(cleaning_report)

# Use the saved cleaned CSV for all analysis below.
df = load_data("outputs\\measurements_clean.csv")
print("Loaded cleaned data:", df.shape)
print("Rows, Columns :", df.shape)

# The dataset has 31 columns - we only need a few for this project.
df = df[["receiver", "tx_latitude", "tx_longitude",
         "rx_latitude", "rx_longitude",
         "distance_m", "rssi1", "rssi2", "rssi",
         "nb_rays", "rx_sensitivity",
         "rssi1_outlier"]].copy()

# =========================================================
# STEP 2: CLEAN / PREPARE
# =========================================================
print("\nMissing values:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))

# The cleaned CSV already contains the average signal in its rssi column.

# =========================================================
# STEP 3: LATITUDE/LONGITUDE  ->  METRES (x, y floor plan)
# =========================================================
# The router (transmitter) is placed at (0, 0).
# 1 degree of latitude is about 111,320 metres.
# Longitude degrees get shorter away from the equator, so multiply by cos(latitude).
router_lat = df["tx_latitude"].iloc[0]
router_lon = df["tx_longitude"].iloc[0]

df["y"] = (df["rx_latitude"] - router_lat) * 111_320
df["x"] = (df["rx_longitude"] - router_lon) * 111_320 * np.cos(np.radians(router_lat))

# =========================================================
# STEP 4: BASIC STATISTICS
# =========================================================
print("\n--- Signal strength (dBm) ---")
print(df["rssi"].describe().round(2))

best = df.loc[df["rssi"].idxmax()]
worst = df.loc[df["rssi"].idxmin()]
print(f"\nStrongest point: {best['receiver']}  {best['rssi']:.1f} dBm  ({best['distance_m']:.1f} m from router)")
print(f"Weakest point  : {worst['receiver']}  {worst['rssi']:.1f} dBm  ({worst['distance_m']:.1f} m from router)")

# =========================================================
# STEP 5: SIGNAL QUALITY CATEGORIES
# =========================================================
# Closer to 0 dBm = stronger signal. Example: -40 is better than -80.
conditions = [
    df["rssi"] >= -50,
    df["rssi"] >= -60,
    df["rssi"] >= -70,
    df["rssi"] >= -80,
]
labels = ["Excellent", "Good", "Fair", "Weak"]
df["quality"] = np.select(conditions, labels, default="Very Weak")

quality_order = ["Excellent", "Good", "Fair", "Weak", "Very Weak"]
quality_count = df["quality"].value_counts().reindex(quality_order, fill_value=0)
quality_pct = (quality_count / len(df) * 100).round(1)
quality_table = pd.DataFrame({"points": quality_count, "percent": quality_pct})
print("\n--- Quality of coverage ---")
print(quality_table)

# =========================================================
# STEP 6: DISTANCE ZONES (groupby)
# =========================================================
bins = [0, 10, 20, 30, 40, 50, 60, 70]
zone_names = ["0-10m", "10-20m", "20-30m", "30-40m", "40-50m", "50-60m", "60-70m"]
df["zone"] = pd.cut(df["distance_m"], bins=bins, labels=zone_names)

zone_table = df.groupby("zone", observed=True)["rssi"].agg(
    points="count", mean="mean", min="min", max="max").round(2)
print("\n--- Signal by distance zone ---")
print(zone_table)

# =========================================================
# STEP 7: DOES DISTANCE REALLY DECIDE THE SIGNAL?
# =========================================================
corr_distance = np.corrcoef(df["distance_m"], df["rssi"])[0, 1]
corr_rays = np.corrcoef(df["nb_rays"], df["rssi"])[0, 1]
print(f"\nCorrelation  distance vs signal : {corr_distance:.2f}")
print(f"Correlation  no. of rays vs signal: {corr_rays:.2f}")
print("(Close to 0 = no relation, close to -1 or +1 = strong relation)")

# =========================================================
# STEP 8: DEAD ZONES AND OUTLIERS
# =========================================================
dead = df[df["rssi"] < df["rx_sensitivity"]]
print(f"\nDead-zone points (below receiver sensitivity): {len(dead)}")
print(dead[["receiver", "distance_m", "rssi"]].round(2).to_string(index=False))
print("Outlier readings flagged in data:", int(df["rssi1_outlier"].sum()))

# =========================================================
# STEP 9: GRID  (floor plan split into 2m x 2m squares)
# =========================================================
cell = 2                                       # square size in metres
df["gx"] = (np.floor(df["x"] / cell) * cell).astype(int)
df["gy"] = (np.floor(df["y"] / cell) * cell).astype(int)

# average signal in each square -> table with rows = y, columns = x
grid = df.pivot_table(index="gy", columns="gx", values="rssi", aggfunc="mean")
grid = grid.sort_index(ascending=False)        # top of the table = north
print("\nGrid size (rows x columns):", grid.shape)

# =========================================================
# STEP 10: HEATMAP IN THE TERMINAL (text map)
# =========================================================
# E = Excellent, G = Good, F = Fair, W = Weak, V = Very Weak,
# . = no measurement, R = router
def letter(value):
    if np.isnan(value):
        return "."
    if value >= -50:
        return "E"
    if value >= -60:
        return "G"
    if value >= -70:
        return "F"
    if value >= -80:
        return "W"
    return "V"

text_map = []
for gy in grid.index:
    row = ""
    for gx in grid.columns:
        if gx == 0 and gy == 0:
            row += "R"
        else:
            row += letter(grid.loc[gy, gx])
    text_map.append(row)

print("\n--- Text heatmap (north is up) ---")
print("\n".join(text_map))

# =========================================================
# STEP 11: COLOUR HEATMAP (HTML page built with NumPy + Pandas)
# =========================================================
# Colour rule: -90 dBm (weak) = red, -65 = yellow, -40 (strong) = green.
def colour(value):
    t = np.clip((value + 90) / 50, 0, 1)                  # 0 to 1
    r = np.interp(t, [0, 0.5, 1], [215, 254, 26])
    g = np.interp(t, [0, 0.5, 1], [48, 224, 152])
    b = np.interp(t, [0, 0.5, 1], [39, 139, 80])
    return f"rgb({int(r)},{int(g)},{int(b)})"

rows_html = ""
for gy in grid.index:
    rows_html += "<tr>"
    for gx in grid.columns:
        value = grid.loc[gy, gx]
        if np.isnan(value):
            bg = "#f2f2f2"
            tip = "no data"
        else:
            bg = colour(value)
            tip = f"{value:.1f} dBm"
        mark = "&#9733;" if (gx == 0 and gy == 0) else ""     # star = router
        rows_html += f'<td style="background:{bg}" title="x={gx}m, y={gy}m: {tip}">{mark}</td>'
    rows_html += "</tr>\n"

# simple bar rows for distance zones (width = how strong the signal is)
bars_html = ""
for name, row in zone_table.iterrows():
    width = int(np.clip((row["mean"] + 90) / 50, 0, 1) * 300)
    bars_html += (f'<div>{name:>8} <span class="bar" style="width:{width}px;'
                  f'background:{colour(row["mean"])}"></span> {row["mean"]} dBm</div>\n')

page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>WiFiScope</title>
<style>
 body {{ font-family: Arial, sans-serif; margin: 20px; }}
 table.map {{ border-collapse: collapse; }}
 table.map td {{ width: 10px; height: 10px; padding: 0; font-size: 9px;
                text-align: center; line-height: 10px; }}
 .bar {{ display: inline-block; height: 14px; vertical-align: middle; }}
 table.data {{ border-collapse: collapse; }}
 table.data td, table.data th {{ border: 1px solid #999; padding: 4px 10px; }}
</style></head><body>
<h1>WiFiScope - Wi-Fi Heatmap Floor Plan</h1>
<p>Each square is 2 m x 2 m. Green = strong signal, red = weak signal,
grey = no measurement, star = router. Hover over a square to see its value.</p>
<table class="map">
{rows_html}</table>
<h2>Coverage quality</h2>
{quality_table.to_html(classes="data")}
<h2>Average signal by distance from router</h2>
{bars_html}
<h2>Key numbers</h2>
<ul>
<li>Total points: {len(df)}</li>
<li>Average signal: {df['rssi'].mean():.1f} dBm</li>
<li>Strongest: {best['rssi']:.1f} dBm ({best['receiver']})</li>
<li>Weakest: {worst['rssi']:.1f} dBm ({worst['receiver']})</li>
<li>Dead-zone points: {len(dead)}</li>
<li>Outlier readings flagged and kept: {int(df['rssi1_outlier'].sum())}</li>
<li>Correlation distance vs signal: {corr_distance:.2f}</li>
<li>Correlation rays vs signal: {corr_rays:.2f}</li>
</ul>
</body></html>"""

with open("outputs\\wifiscope_heatmap.html", "w", encoding="utf-8") as f:
    f.write(page)

# =========================================================
# STEP 12: SAVE RESULTS
# =========================================================
df.to_csv("outputs\\wifiscope_results.csv", index=False)
zone_table.to_csv("outputs\\wifiscope_zone_summary.csv")

with open("outputs\\wifiscope_summary_report.txt", "w", encoding="utf-8") as f:
    f.write("WIFISCOPE - SUMMARY REPORT\n")
    f.write("=" * 40 + "\n")
    f.write(f"Total measurement points : {len(df)}\n")
    f.write(f"Average signal           : {df['rssi'].mean():.1f} dBm\n")
    f.write(f"Strongest                : {best['rssi']:.1f} dBm ({best['receiver']})\n")
    f.write(f"Weakest                  : {worst['rssi']:.1f} dBm ({worst['receiver']})\n")
    f.write(f"Dead-zone points         : {len(dead)}\n")
    f.write(f"Outlier readings flagged : {int(df['rssi1_outlier'].sum())} (kept)\n")
    f.write(f"Distance-signal corr.    : {corr_distance:.2f}\n")
    f.write(f"Rays-signal corr.        : {corr_rays:.2f}\n\n")
    f.write("Coverage quality:\n")
    f.write(quality_table.to_string())
    f.write("\n\nSignal by distance zone:\n")
    f.write(zone_table.to_string())
    f.write("\n\nText heatmap (north is up):\n")
    f.write("\n".join(text_map))

print("\nDone! Open outputs\\wifiscope_heatmap.html in your browser to see the heatmap.")
