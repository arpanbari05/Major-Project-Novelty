import h5py
import numpy as np
import pandas as pd
import os
from datetime import datetime

# -------------------------------
# Utility functions
# -------------------------------
def compute_relative_azimuth(sun_az, sat_az):
    """Compute Relative Azimuth Angle (RAA) = |Sun_Azimuth - Sat_Azimuth| adjusted to [0, 180]."""
    raa = np.abs(sun_az - sat_az)
    raa = np.where(raa > 180, 360 - raa, raa)
    return raa

def clean_fill(arr, fill_value=999):
    """Replace fill values with NaN."""
    arr = np.where(arr == fill_value, np.nan, arr)
    return arr

def match_lengths(arrays: dict):
    """Trim all arrays to the same minimum length."""
    min_len = min(len(v) for v in arrays.values())
    return {k: v[:min_len] for k, v in arrays.items()}

def extract_datetime_from_filename(filename: str):
    """
    Extract date (dd:mm:yyyy) and time (hh:mm:ss) from filename.
    Example: 3RIMG_19OCT2025_0015_L1C_SGP_V01R00.h5 -> ('19:10:2025', '00:15:00')
    """
    try:
        parts = filename.split("_")
        date_str = parts[1]  # e.g., 19OCT2025
        time_str = parts[2]  # e.g., 0015
        date = datetime.strptime(date_str, "%d%b%Y").strftime("%d:%m:%Y")
        hour = int(time_str[:2])
        minute = int(time_str[2:])
        time = f"{hour:02d}:{minute:02d}:00"
        return date, time
    except Exception as e:
        print(f"⚠️ Warning: Could not parse datetime from {filename}: {e}")
        return None, None

# -------------------------------
# Main HDF5 feature extraction
# -------------------------------
def load_h5_features(filepath: str):
    """Extract key features from an HDF5 satellite file."""
    with h5py.File(filepath, "r") as f:
        arrays = {}

        # --- Radiances ---
        arrays["IMG_VIS_RADIANCE"] = clean_fill(np.array(f["IMG_VIS_RADIANCE"]).astype(np.float32).flatten())
        arrays["IMG_SWIR_RADIANCE"] = clean_fill(np.array(f["IMG_SWIR_RADIANCE"]).astype(np.float32).flatten())
        arrays["IMG_MIR_RADIANCE"] = clean_fill(np.array(f["IMG_MIR_RADIANCE"]).astype(np.float32).flatten())
        arrays["IMG_TIR1_RADIANCE"] = clean_fill(np.array(f["IMG_TIR1_RADIANCE"]).astype(np.float32).flatten())
        arrays["IMG_TIR2_RADIANCE"] = clean_fill(np.array(f["IMG_TIR2_RADIANCE"]).astype(np.float32).flatten())
        arrays["IMG_WV_RADIANCE"] = clean_fill(np.array(f["IMG_WV_RADIANCE"]).astype(np.float32).flatten())

        # --- Reflectance / Albedo ---
        arrays["IMG_VIS_ALBEDO"] = clean_fill(np.array(f["IMG_VIS_ALBEDO"]).astype(np.float32).flatten())
        arrays["IMG_SWIR"] = clean_fill(np.array(f["IMG_SWIR"]).astype(np.float32).flatten())

        # --- Brightness Temps ---
        arrays["IMG_MIR_TEMP"] = np.array(f["IMG_MIR_TEMP"]).astype(np.float32).flatten()
        arrays["IMG_TIR1_TEMP"] = np.array(f["IMG_TIR1_TEMP"]).astype(np.float32).flatten()
        arrays["IMG_TIR2_TEMP"] = np.array(f["IMG_TIR2_TEMP"]).astype(np.float32).flatten()
        arrays["IMG_WV_TEMP"] = np.array(f["IMG_WV_TEMP"]).astype(np.float32).flatten()

        # --- Geometry ---
        arrays["Sun_Azimuth"] = np.array(f["Sun_Azimuth"]).astype(np.float32).flatten() * 0.01
        arrays["Sun_Elevation"] = np.array(f["Sun_Elevation"]).astype(np.float32).flatten() * 0.01
        arrays["Sat_Azimuth"] = np.array(f["Sat_Azimuth"]).astype(np.float32).flatten() * 0.01
        arrays["Sat_Elevation"] = np.array(f["Sat_Elevation"]).astype(np.float32).flatten() * 0.01

        # --- Geolocation ---
        arrays["X"] = np.array(f["X"]).astype(np.float32).flatten()
        arrays["Y"] = np.array(f["Y"]).astype(np.float32).flatten()

        # ✅ Trim to consistent length
        arrays = match_lengths(arrays)

        # --- Derived Angles ---
        solar_zenith = 90.0 - arrays["Sun_Elevation"]
        viewing_zenith = 90.0 - arrays["Sat_Elevation"]
        rel_azimuth = compute_relative_azimuth(arrays["Sun_Azimuth"], arrays["Sat_Azimuth"])

        df = pd.DataFrame({
            "IMG_VIS_ALBEDO": arrays["IMG_VIS_ALBEDO"],
            "IMG_SWIR": arrays["IMG_SWIR"],
            "IMG_VIS_RADIANCE": arrays["IMG_VIS_RADIANCE"],
            "IMG_SWIR_RADIANCE": arrays["IMG_SWIR_RADIANCE"],
            "IMG_MIR_RADIANCE": arrays["IMG_MIR_RADIANCE"],
            "IMG_TIR1_RADIANCE": arrays["IMG_TIR1_RADIANCE"],
            "IMG_TIR2_RADIANCE": arrays["IMG_TIR2_RADIANCE"],
            "IMG_WV_RADIANCE": arrays["IMG_WV_RADIANCE"],
            "IMG_MIR_TEMP": arrays["IMG_MIR_TEMP"],
            "IMG_TIR1_TEMP": arrays["IMG_TIR1_TEMP"],
            "IMG_TIR2_TEMP": arrays["IMG_TIR2_TEMP"],
            "IMG_WV_TEMP": arrays["IMG_WV_TEMP"],
            "Sun_Azimuth": arrays["Sun_Azimuth"],
            "Sun_Elevation": arrays["Sun_Elevation"],
            "Solar_Zenith": solar_zenith,
            "Sat_Azimuth": arrays["Sat_Azimuth"],
            "Sat_Elevation": arrays["Sat_Elevation"],
            "Viewing_Zenith": viewing_zenith,
            "Relative_Azimuth": rel_azimuth,
            "X": arrays["X"],
            "Y": arrays["Y"],
        })

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        return df

# -------------------------------
# Folder-level processing
# -------------------------------
def process_all_h5(folder_path="Datasets"):
    """Process all .h5 files and combine into a single CSV."""
    all_dfs = []
    for file in os.listdir(folder_path):
        if file.endswith(".h5"):
            filepath = os.path.join(folder_path, file)
            print(f"📂 Processing {file} ...")

            df = load_h5_features(filepath)
            date, time = extract_datetime_from_filename(file)
            df["Date_(dd:mm:yyyy)"] = date
            df["Time_(hh:mm:ss)"] = time
            all_dfs.append(df)

    if not all_dfs:
        raise ValueError("No .h5 files found in the Datasets folder!")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv("aod_dataset.csv", index=False)
    print(f"\n✅ Saved combined dataset as 'aod_dataset.csv' (shape={combined.shape})")


if __name__ == "__main__":
    process_all_h5("Datasets")
