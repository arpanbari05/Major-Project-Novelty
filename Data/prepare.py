import h5py
import numpy as np
import pandas as pd

def compute_relative_azimuth(sun_az, sat_az):
    """Compute Relative Azimuth Angle (RAA) = |Sun_Azimuth - Sat_Azimuth|, adjusted to [0, 180]."""
    raa = np.abs(sun_az - sat_az)
    raa = np.where(raa > 180, 360 - raa, raa)
    return raa

def clean_fill(arr, fill_value=999):
    """Replace fill values (like 999) with NaN."""
    arr = np.where(arr == fill_value, np.nan, arr)
    return arr

def match_lengths(arrays: dict):
    """Trim all arrays to the same minimum length."""
    lengths = {k: len(v) for k, v in arrays.items()}
    min_len = min(lengths.values())
    print("Detected array lengths:", lengths)
    print(f"⚙️ Trimming all arrays to minimum length = {min_len}")
    return {k: v[:min_len] for k, v in arrays.items()}

def load_h5_features(filepath: str):
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

        # ✅ Trim all arrays to same minimum length
        arrays = match_lengths(arrays)

        # --- Derived Angles ---
        solar_zenith = 90.0 - arrays["Sun_Elevation"]
        viewing_zenith = 90.0 - arrays["Sat_Elevation"]
        rel_azimuth = compute_relative_azimuth(arrays["Sun_Azimuth"], arrays["Sat_Azimuth"])

        # --- AOD placeholder target ---
        aod_target = np.random.uniform(0.1, 1.0, size=len(arrays["X"])).astype(np.float32)

        # --- Build DataFrame ---
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
            "AOD_Target": aod_target
        })

        # --- Clean NaNs ---
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        return df


if __name__ == "__main__":
    filepath = "Datasets/3RIMG_19OCT2025_0015_L1C_SGP_V01R00.H5"  # 🔹 Change to your actual .h5 path
    df = load_h5_features(filepath)
    df.to_csv("aod_dataset.csv", index=False)
    print(f"✅ Saved 'aod_dataset.csv' successfully! Shape = {df.shape}")
