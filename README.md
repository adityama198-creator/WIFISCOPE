# WiFiScope - Wi-Fi Heatmap Floor Plan

WiFiScope reads Wi-Fi measurements, cleans the data, studies signal strength,
and creates a floor-plan heatmap. It uses Python with NumPy and Pandas.

## How to run

1. Keep the original file named `wifiscope.csv` in the `data` folder.
2. Open a terminal in the main WIFISCOPE project folder.
3. Run:

   ```text
   python wifiscope.py
   ```

The program creates its cleaned data and results in the `outputs` folder.

## What happens when the program runs

### 1. Load the original data

The program reads `data/wifiscope.csv`. This file contains the original
measurements, including the receiver positions and two signal readings,
`rssi1` and `rssi2`.

### 2. Clean the measurements

The cleaning step:

- Removes rows missing important values.
- Removes duplicate rows.
- Removes rows with invalid latitude, longitude, or RSSI values.
- Adds a distance from the router, calculated from the transmitter and
  receiver coordinates.

The cleaned data is saved as `outputs/measurements_clean.csv`. The program then
reads this saved file for the rest of the analysis, so the results use the
cleaned data rather than the original file.

### 3. Find and mark outliers

The program uses the IQR rule to identify unusually high or low RSSI readings.
It marks them in the `rssi1_outlier` and `rssi2_outlier` columns.

Flagged readings are kept, not deleted. A very weak reading could be a real
dead spot, so removing it automatically might hide an important part of the
coverage map. The cleaning report records how many readings were flagged.

### 4. Make one signal value

Each receiver has two signal readings. The program averages them to make one
`rssi` value. RSSI is measured in dBm; a value closer to zero, such as -40 dBm,
means a stronger signal than -80 dBm.

### 5. Convert receiver positions to a floor-plan map

The router is treated as the map's `(0, 0)` point. The program converts the
latitude and longitude differences into approximate x and y distances in
metres. This lets the receiver points be placed on a flat grid.

### 6. Summarize signal strength

The program prints the average, minimum, maximum, and other basic statistics.
It also identifies the strongest and weakest receiver points.

### 7. Label signal quality

Every measurement gets a quality label based on its RSSI:

| RSSI | Quality |
| --- | --- |
| -50 dBm or stronger | Excellent |
| -60 dBm or stronger | Good |
| -70 dBm or stronger | Fair |
| -80 dBm or stronger | Weak |
| Weaker than -80 dBm | Very Weak |

The program counts the points in each category and calculates their
percentages.

### 8. Compare distance zones

The program groups receivers into distance bands, such as 0-10 metres and
10-20 metres. It summarizes the number of points and the signal strength in
each band.

### 9. Compare signal with distance and ray count

The program calculates correlations between signal and distance, and between
signal and the number of signal rays. A correlation near zero means little
linear relationship; a value closer to -1 or +1 means a stronger relationship.
Correlation does not prove that one measurement causes another.

### 10. Find dead zones

The receiver sensitivity is the minimum signal level the receiver can use.
Any point with an RSSI below that sensitivity is listed as a dead-zone point.
The program also reports the outlier flags.

### 11. Build the heatmap

The floor plan is divided into 2 m by 2 m squares. If more than one receiver
falls in a square, their RSSI values are averaged.

The program prints a compact terminal dashboard containing the key numbers,
coverage-quality bars, distance-zone signal bars, and a text heatmap. A 2 m by
2 m square is shown as `E`, `G`, `F`, `W`, or `V` for its signal quality.
The router is marked `R`, and `.` means no measurement in that square. It also
creates `outputs/wifiscope_heatmap.html`, a colour heatmap that can be opened in
a web browser. The heatmap adapts to narrow screens and can be scrolled
horizontally when the full map does not fit.

### 12. Save the results

The program saves the cleaned data, cleaning report, analysis results, distance
zone summary, text report, and HTML heatmap in the `outputs` folder.
