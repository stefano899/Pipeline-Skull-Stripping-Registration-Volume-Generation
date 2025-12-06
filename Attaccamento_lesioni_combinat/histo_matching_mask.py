import SimpleITK as sitk

# --- Path ---
lesions_path = r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\BraTS-GLI-00000-000\Registered\volume_lesioni_flair_zscore_normalized_BRATS.nii.gz"
reference_path = r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\sub08(check flair e t1)\volumes_generated_no_bandoni\volume_fake_flair_matched_to_mni.nii.gz"
output_path = r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\sub08(check flair e t1)\volumes_generated_no_bandoni\GLIOMA_mask_volume_zscore_normalized_histo_matched.nii.gz"

# --- Carica immagini come float ---
lesions_img = sitk.ReadImage(lesions_path, sitk.sitkFloat32)
reference_img = sitk.ReadImage(reference_path, sitk.sitkFloat32)

# --- Histogram matching ---
matcher = sitk.HistogramMatchingImageFilter()
matcher.SetNumberOfHistogramLevels(256)
matcher.SetNumberOfMatchPoints(64)
matcher.ThresholdAtMeanIntensityOn()   # riduce l'effetto dello sfondo (0)

matched_lesions = matcher.Execute(lesions_img, reference_img)

# --- Mantieni background = 0 (solo dove ci sono lesioni) ---
# Crea una maschera dalle lesioni originali (voxel > 0)
lesions_mask = sitk.Cast(lesions_img > 0, sitk.sitkFloat32)

matched_lesions_masked = matched_lesions * lesions_mask

# --- Salva risultato ---
sitk.WriteImage(matched_lesions_masked, output_path)

print("Histogram matching completato. File salvato in:")
print(output_path)
