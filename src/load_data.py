
import pandas as pd


def load_data(file_path):
    """Load the original WiFiScope CSV dataset."""

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        ) from error

    if df.empty:
        raise ValueError("The CSV file is empty.")

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df
