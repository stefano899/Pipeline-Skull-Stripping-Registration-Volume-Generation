import nibabel as nib
import numpy as np

# --- Input ---
volume_path = r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\BraTS-GLI-00000-000\Registered\BraTS-GLI-00000-000-t2f.nii_in_MNI_FULL.nii.gz"
mask_path = r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\BraTS-GLI-00000-000\Registered\volume_lesioni\volume_mask_lesioni_SMOOTH_SMOOTH_SMOOTH_FILLED_BIN.nii.gz"
output_path = r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\BraTS-GLI-00000-000\Registered\volume_lesioni\VOLUME_SENZA_TUMORE.nii.gz"

# --- Carica ---
vol_img = nib.load(volume_path)
mask_img = nib.load(mask_path)

vol = vol_img.get_fdata()
vol = vol_img.get_fdata()
mask = mask_img.get_fdata()

# --- Controllo dimensioni ---
if vol.shape != mask.shape:
    raise ValueError(f"Dimensioni incompatibili: volume {vol.shape}, maschera {mask.shape}")

# --- Rendi la maschera binaria (nel caso non lo fosse) ---
mask_bin = (mask != 0).astype(np.uint8)

# --- Inverti la maschera: dentro il tumore = 0, fuori = 1 ---
inv_mask = 1 - mask_bin

# --- Applica l'inversione al volume ---
volume_senza_tumore = vol * inv_mask

# --- Salvataggio ---
output_img = nib.Nifti1Image(volume_senza_tumore, affine=vol_img.affine, header=vol_img.header)
nib.save(output_img, output_path)

print("Creato volume con tumore azzerato in:", output_path)
