import nibabel as nib
import numpy as np
from skimage.filters import threshold_otsu
import nibabel as nib

# --- Carica NIfTI ---
input_path = r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\BraTS-GLI-00000-000\Registered\volume_lesioni\volume_T1_normalized_lesioni.nii.gz"
img = nib.load(input_path)
data = img.get_fdata()

# --- Calcola soglia Otsu ---
# (applicata sull’intero volume)
flat = data[data > 0]        # evita lo sfondo se necessario
th = threshold_otsu(flat)

print("Soglia Otsu:", th)

# --- Applica threshold ---
binary = (data >= th).astype(np.uint8)

# --- Salva come NIfTI ---
output_path = r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\BraTS-GLI-00000-000\Registered\volume_lesioni\volume_mask_lesioni.nii.gz"
new_img = nib.Nifti1Image(binary, affine=img.affine, header=img.header)
nib.save(new_img, output_path)

print("File salvato in:", output_path)
