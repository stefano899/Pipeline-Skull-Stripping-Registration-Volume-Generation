import SimpleITK as sitk

# --- Volumi di lesione da smussare e "riempire" ---
lesion_volumes_paths = [
    r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\testing_08_patient_07 (per lesioni)\Testing_Center_08_Patient_07_Consensus_SMOOTH_SMOOTH.nii.gz",
    r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\BraTS-GLI-00000-000\Registered\volume_lesioni\volume_mask_lesioni_SMOOTH_SMOOTH.nii.gz",
]

# Parametro smoothing
sigma = 1.5   # puoi usare 1.0, 1.5, 2.0…

for lesion_path in lesion_volumes_paths:
    print("Smoothing + fill holes + threshold per volume:", lesion_path)

    # --- 1) Carica volume lesione ---
    lesion_img = sitk.ReadImage(lesion_path, sitk.sitkFloat32)

    # --- 2) Smooth gaussiano del volume di lesioni ---
    smooth_raw = sitk.SmoothingRecursiveGaussian(lesion_img, sigma)

    # --- 3) Crea una maschera dai voxel > 0 dello smooth ---
    smooth_mask = smooth_raw > 0

    # --- 4) Riempimento dei buchini nella maschera ---
    filled_mask = sitk.BinaryFillhole(smooth_mask, fullyConnected=True)

    # --- 5) Applica la maschera riempita allo smoothing ---
    filled_mask_f = sitk.Cast(filled_mask, sitk.sitkFloat32)
    lesion_smooth_filled = smooth_raw * filled_mask_f

    # --- 6) Threshold: se <0.4 diventa 0, se ≥0.4 diventa 1 ---
    threshold_value = 0.25
    lesion_thresholded = lesion_smooth_filled >= threshold_value
    lesion_thresholded = sitk.Cast(lesion_thresholded, sitk.sitkUInt8)

    # --- 7) Salva il volume finale binarizzato ---
    output_path = lesion_path.replace(".nii", "_SMOOTH_FILLED_BIN.nii")
    sitk.WriteImage(lesion_thresholded, output_path)

    print(" --> Salvato:", output_path)

print("\nFINITO! Volume smooth → filled → thresholded.")
