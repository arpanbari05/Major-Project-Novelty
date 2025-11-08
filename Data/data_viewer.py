import h5py
import numpy as np

path = "Datasets/3RIMG_19OCT2025_0015_L1C_SGP_V01R00.H5"

with h5py.File(path, 'r') as f:
    # list keys
    print("Available keys:\n", list(f.keys()))

    # choose a few important variables
    keys_to_check = [
        'IMG_VIS_ALBEDO',
        'IMG_VIS_RADIANCE',
        'IMG_SWIR_RADIANCE',
        'IMG_TIR1_TEMP',
        'Sun_Elevation',
        'Sat_Elevation'
    ]

    for k in f.keys():
        try:
            print(f"{k:25s}  shape={f[k].shape}")
        except:
            pass

    print("\n--- Sample values (top-left 10 elements along first row) ---\n")
    for key in keys_to_check:
        if key in f:
            data = f[key][:]
            print(f"{key}: shape={data.shape}, dtype={data.dtype}")
            # print first 10 pixel values from first row
            print("First 10 values:", data[:10])
            # also check dataset attributes (often contain units, scale factors)
            attrs = {k: v for k, v in f[key].attrs.items()}
            print("Attributes:", attrs, "\n")
        else:
            print(f"{key}: not found in this file\n")
        
    print(f['IMG_VIS'].attrs)
    print(f['IMG_SWIR'].attrs)

