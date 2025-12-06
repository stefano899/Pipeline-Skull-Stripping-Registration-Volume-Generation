import nibabel as nib
import numpy as np

# --- Input ---
ref_path = r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\testing_08_patient_07 (per lesioni)\VOLUME_SENZA_TUMORE.nii.gz"
target_path = r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\testing_08_patient_07 (per lesioni)\Testing_Center_08_Patient_07_FLAIR_preprocessed_inMNI_FULL.nii.gz"

output_path = r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\testing_08_patient_07 (per lesioni)\VOLUME_TUMORE_MS_Z-SCORE.nii.gz"

# --- Carico volumi ---
ref_img = nib.load(ref_path)
target_img = nib.load(target_path)

ref = ref_img.get_fdata().astype(np.float32)
target = target_img.get_fdata().astype(np.float32)

# --- Stima mean/std su voxel NON-zero del riferimento ---
ref_nonzero = ref[ref != 0]

if len(ref_nonzero) == 0:
    raise ValueError("Il volume di riferimento non contiene voxel > 0. Impossibile calcolare mean/std.")

ref_mean = ref_nonzero.mean()
ref_std = ref_nonzero.std()

print("MEDIA del volume riferimento:", ref_mean)
print("STD del volume riferimento:", ref_std)

# --- Calcolo Z-score del target ---
target_zscore = np.zeros_like(target)
target_nonzero = target[target != 0]

if ref_std == 0:
    raise ValueError("La deviazione standard del volume di riferimento è zero. Normalizzazione impossibile.")

target_zscore[target != 0] = (target_nonzero - ref_mean) / ref_std

# ───────────────────────────────────────────
# 🔵 NORMALIZZAZIONE 0–1 (min–max) SOLO SUI NON-ZERO
# ───────────────────────────────────────────

z_nonzero = target_zscore[target_zscore != 0]

if len(z_nonzero) == 0:
    raise ValueError("Nessun voxel non-zero dopo Z-score. Impossibile fare min-max.")

z_min = z_nonzero.min()
z_max = z_nonzero.max()

target_norm = np.zeros_like(target_zscore)
target_norm[target_zscore != 0] = (target_zscore[target_zscore != 0] - z_min) / (z_max - z_min)

# --- Salvataggio ---
norm_img = nib.Nifti1Image(target_norm, affine=target_img.affine, header=target_img.header)
nib.save(norm_img, output_path)

print("Z-score + normalizzazione 0–1 salvato in:", output_path)
