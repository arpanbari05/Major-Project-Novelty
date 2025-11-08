# ============================================================
# Script: extract_aeronet_data.py
# Purpose: Convert all Aeronet .lev20 / .ONEILL_lev20 files → CSV
# ============================================================

import pandas as pd
from pathlib import Path
import numpy as np

def load_single_aeronet_file(file_path: Path):
    """
    Loads an Aeronet Level 2.0 file, skipping headers and extracting valid columns.
    Expected columns:
        - Date_(dd:mm:yyyy)
        - Time_(hh:mm:ss)
        - Angstrom_Exponent(AE)-Total_500nm[alpha]
        - FineModeFraction_500nm[eta]
        - Total_AOD_500nm[tau_a]
    """
    try:
        df = pd.read_csv(file_path, skiprows=6)
    except Exception as e:
        print(f"⚠️ Could not read {file_path.name}: {e}")
        return None

    # Try to detect the available column names (some files have slightly different headers)
    cols = list(df.columns)
    target_cols = {
        "Date_(dd:mm:yyyy)": None,
        "Time_(hh:mm:ss)": None,
        "Angstrom_Exponent(AE)-Total_500nm[alpha]": None,
        "FineModeFraction_500nm[eta]": None,
        "Total_AOD_500nm[tau_a]": None,
    }

    for c in cols:
        for key in target_cols:
            if key.lower().replace("_", "").replace("-", "") in c.lower().replace("_", "").replace("-", ""):
                target_cols[key] = c

    # Check if all were found
    if any(v is None for v in target_cols.values()):
        missing = [k for k, v in target_cols.items() if v is None]
        print(f"⚠️ Skipped {file_path.name}: Missing columns {missing}")
        return None

    df = df[list(target_cols.values())]
    df.columns = list(target_cols.keys())  # Rename to standard form

    # Clean fill values
    df.replace(-999.0, np.nan, inplace=True)
    df.dropna(inplace=True)

    # Parse datetime
    df["datetime"] = pd.to_datetime(
        df["Date_(dd:mm:yyyy)"] + " " + df["Time_(hh:mm:ss)"],
        errors="coerce",
        format="%d:%m:%Y %H:%M:%S"
    )

    df.dropna(subset=["datetime"], inplace=True)
    df = df[[
        "datetime",
        "Angstrom_Exponent(AE)-Total_500nm[alpha]",
        "FineModeFraction_500nm[eta]",
        "Total_AOD_500nm[tau_a]"
    ]]

    df.rename(columns={
        "Angstrom_Exponent(AE)-Total_500nm[alpha]": "AE_500nm_alpha",
        "FineModeFraction_500nm[eta]": "FMF_500nm_eta",
        "Total_AOD_500nm[tau_a]": "AOD_500nm_tau_a"
    }, inplace=True)

    print(f"✅ Processed {file_path.name}, shape={df.shape}")
    return df


def load_all_aeronet_data(folder_path: str):
    """Combine all Aeronet files in the folder into one CSV."""
    folder = Path(folder_path)
    files = list(folder.glob("*.lev20")) + list(folder.glob("*.ONEILL_lev20"))
    if not files:
        raise ValueError("No valid Aeronet .lev20 or .ONEILL_lev20 files found in folder.")

    dfs = []
    for f in files:
        df = load_single_aeronet_file(f)
        if df is not None and not df.empty:
            dfs.append(df)

    if not dfs:
        raise ValueError("No valid data could be extracted from Aeronet files.")

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.drop_duplicates(subset=["datetime"], inplace=True)
    combined_df.sort_values("datetime", inplace=True)
    return combined_df


if __name__ == "__main__":
    aeronet_folder = "Aeronet"
    aeronet_df = load_all_aeronet_data(aeronet_folder)
    aeronet_df.to_csv("aeronet_dataset.csv", index=False)
    print(f"\n✅ Saved 'aeronet_dataset.csv' successfully! Shape = {aeronet_df.shape}")
