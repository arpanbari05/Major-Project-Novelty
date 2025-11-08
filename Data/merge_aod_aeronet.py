import pandas as pd
from datetime import datetime, timedelta

# ---------------------------------------
# Utility
# ---------------------------------------
def parse_datetime(date_str, time_str):
    """Convert 'dd:mm:yyyy' + 'hh:mm:ss' to datetime."""
    try:
        return datetime.strptime(f"{date_str} {time_str}", "%d:%m:%Y %H:%M:%S")
    except Exception:
        return None


def load_satellite_data(csv_path):
    """Load satellite (AOD) dataset."""
    df = pd.read_csv(csv_path)
    if "Date_(dd:mm:yyyy)" not in df.columns or "Time_(hh:mm:ss)" not in df.columns:
        raise ValueError("Missing date/time columns in satellite dataset.")
    
    # Convert to datetime
    df["datetime"] = df.apply(lambda r: parse_datetime(r["Date_(dd:mm:yyyy)"], r["Time_(hh:mm:ss)"]), axis=1)
    df.dropna(subset=["datetime"], inplace=True)
    print(f"✅ Satellite data loaded: {df.shape[0]} rows")
    return df


def load_aeronet_data(csv_path):
    """Load Aeronet dataset (already preprocessed)."""
    df = pd.read_csv(csv_path)
    if "datetime" not in df.columns:
        raise ValueError("Missing 'datetime' column in Aeronet dataset.")
    
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df.dropna(subset=["datetime"], inplace=True)
    print(f"✅ Aeronet data loaded: {df.shape[0]} rows")
    return df


def merge_with_tolerance(sat_df, aeronet_df, tolerance_minutes=30):
    """Merge satellite and Aeronet datasets within ±tolerance minutes."""
    tolerance = timedelta(minutes=tolerance_minutes)
    merged_rows = []

    sat_df = sat_df.sort_values("datetime").reset_index(drop=True)
    aeronet_df = aeronet_df.sort_values("datetime").reset_index(drop=True)

    a_idx = 0
    for _, s_row in sat_df.iterrows():
        s_time = s_row["datetime"]

        # Find nearest Aeronet record (simple linear scan)
        while a_idx < len(aeronet_df) - 1 and aeronet_df.loc[a_idx + 1, "datetime"] < s_time:
            a_idx += 1

        nearest = aeronet_df.loc[a_idx]
        diff = abs((nearest["datetime"] - s_time).total_seconds())

        if diff <= tolerance.total_seconds():
            merged = {**s_row.to_dict(), **nearest.to_dict()}
            merged_rows.append(merged)

    merged_df = pd.DataFrame(merged_rows)
    print(f"✅ Merged dataset created: {merged_df.shape[0]} matched rows (tolerance ±{tolerance_minutes} min)")
    return merged_df


# ---------------------------------------
# Main
# ---------------------------------------
if __name__ == "__main__":
    aod_path = "aod_dataset.csv"
    aeronet_path = "aeronet_dataset.csv"
    output_path = "merged_aod_aeronet.csv"

    print("🚀 Loading datasets ...")
    sat_df = load_satellite_data(aod_path)
    aeronet_df = load_aeronet_data(aeronet_path)

    print("🔄 Merging datasets with ±30 min tolerance ...")
    merged_df = merge_with_tolerance(sat_df, aeronet_df, tolerance_minutes=30)

    merged_df.to_csv(output_path, index=False)
    print(f"\n✅ Saved merged dataset: {output_path}")
