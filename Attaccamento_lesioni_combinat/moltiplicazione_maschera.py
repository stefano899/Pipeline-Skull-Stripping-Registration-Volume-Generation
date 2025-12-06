import nibabel as nib
import numpy as np

# --- Input ---
volume_path = r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\testing_08_patient_07 (per lesioni)\Testing_Center_08_Patient_07_FLAIR_preprocessed_inMNI_FULL_HISTO_MATCHED.nii.gz"
mask_path   = r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\testing_08_patient_07 (per lesioni)\Testing_Center_08_Patient_07_Consensus_SMOOTH_SMOOTH_SMOOTH_FILLED_BIN.nii.gz"
output_path = r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\testing_08_patient_07 (per lesioni)\Testing_Center_08_Patient_07_FLAIR_preprocessed_inMNI_FULL_HISTO_MATCHED_VOLUME_LESIONI.nii.gz"

# --- Carica ---
vol_img = nib.load(volume_path)
mask_img = nib.load(mask_path)

vol  = vol_img.get_fdata()
mask = mask_img.get_fdata()

# --- Controllo dimensioni ---
if vol.shape != mask.shape:
    raise ValueError(f"Dimensioni incompatibili: volume {vol.shape}, maschera {mask.shape}")

# --- Assicuro maschera binaria ---
mask_bin = (mask > 0).astype(np.uint8)

# --- Estrazione lesioni (solo moltiplicazione) ---
lesions = vol * mask_bin

# --- Salvataggio ---
lesions_img = nib.Nifti1Image(lesions, affine=vol_img.affine, header=vol_img.header)
nib.save(lesions_img, output_path)

print("Volume con sole lesioni salvato in:", output_path)
