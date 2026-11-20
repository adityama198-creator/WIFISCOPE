
import pandas as pd
import numpy as np


def find_column(df, prefix):
    """Find a column using its starting name."""
    matches = [
        col for col in df.columns
        if col.startswith(prefix)
    ]

    return matches[0] if matches else None


def calculate_distance(df):
    """Calculate geographic distance using coordinates."""

    lat1 = np.radians(df["tx_latitude"])
    lon1 = np.radians(df["tx_longitude"])

    lat2 = np.radians(df["rx_latitude"])
    lon2 = np.radians(df["rx_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    a = np.clip(a, 0, 1)

    distance = (
        6371000
        * 2
        * np.arcsin(np.sqrt(a))
    )

    return distance


def clean_data(df):

    original_rows = len(df)
    report = []

    report.append("WIFISCOPE DATA CLEANING REPORT")
    report.append("=" * 40)

    report.append(f"Original rows: {original_rows}")

    # Identify important columns
    required = [
        "receiver",
        "tx_latitude",
        "tx_longitude",
        "rx_latitude",
        "rx_longitude",
        "rssi1",
        "rssi2"
    ]

    missing_columns = [
        col for col in required
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # Convert numerical columns
    numeric_columns = [
        "tx_latitude",
        "tx_longitude",
        "rx_latitude",
        "rx_longitude",
        "rssi1",
        "rssi2"
    ]

    for col in df.columns:
        if col in numeric_columns:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )

    # Missing values
    missing_before = int(df.isnull().sum().sum())

    # Remove rows missing essential measurements
    df = df.dropna(subset=required)

    removed_missing = original_rows - len(df)

    report.append(
        f"Rows removed due to missing essential values: "
        f"{removed_missing}"
    )

    # Remove duplicate records
    duplicates = int(df.duplicated().sum())

    df = df.drop_duplicates()

    report.append(
        f"Duplicate rows removed: {duplicates}"
    )

    # Validate geographic coordinates
    coordinate_valid = (
        df["tx_latitude"].between(-90, 90)
        & df["rx_latitude"].between(-90, 90)
        & df["tx_longitude"].between(-180, 180)
        & df["rx_longitude"].between(-180, 180)
    )

    invalid_coordinates = int(
        (~coordinate_valid).sum()
    )

    df = df[coordinate_valid].copy()

    report.append(
        f"Invalid coordinate rows removed: "
        f"{invalid_coordinates}"
    )

    # Validate RSSI measurements
    for col in ["rssi1", "rssi2"]:

        valid_rssi = df[col].between(-100, -20)

        invalid_count = int((~valid_rssi).sum())

        df = df[valid_rssi].copy()

        report.append(
            f"{col}: invalid RSSI readings removed: "
            f"{invalid_count}"
        )

    # Outlier detection using IQR
    for col in ["rssi1", "rssi2"]:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_col = col + "_outlier"

        df[outlier_col] = (
            (df[col] < lower)
            | (df[col] > upper)
        )

        report.append(
            f"{col} outliers detected: "
            f"{int(df[outlier_col].sum())}"
        )

    # Keep outliers because very weak readings may be real dead spots.
    # The flags let later steps identify these readings without hiding them.
    df["rssi"] = (df["rssi1"] + df["rssi2"]) / 2
    report.append(
        "Outlier readings were flagged and kept in the cleaned data."
    )

    # Calculate distance in metres
    df["distance_m"] = calculate_distance(df)

    # Calculate logarithmic distance
    df["log_distance"] = np.where(
        df["distance_m"] > 0,
        np.log10(df["distance_m"]),
        np.nan
    )

    report.append(
        "Distance and logarithmic distance calculated."
    )

    # Minimum valid row requirement
    if len(df) < 10:
        raise ValueError(
            "Fewer than 10 valid records remain."
        )

    # Final report
    report.append("")
    report.append("FINAL SUMMARY")
    report.append("-" * 40)

    report.append(
        f"Final valid rows: {len(df)}"
    )

    report.append(
        f"Rows removed: {original_rows - len(df)}"
    )

    report.append(
        f"Missing values remaining: "
        f"{int(df.isnull().sum().sum())}"
    )

    return df, "\n".join(report)
