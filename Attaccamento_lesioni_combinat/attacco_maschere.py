import SimpleITK as sitk
import numpy as np


# ==========================================================
# PERCORSI FILE
# ==========================================================
reference_path = r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\sub08(check flair e t1)\volumes_generated_no_bandoni\output_overlay\volume_FLAIR_histo_con_tumore.nii.gz"

lesion_volumes_paths = [
    r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\sub08(check flair e t1)\volumes_generated_no_bandoni\output_overlay\Volume_solo_lesioni_GLIOMA.nii.gz",
    r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\sub08(check flair e t1)\volumes_generated_no_bandoni\output_overlay\Testing_Center_08_Patient_07_FLAIR_preprocessed_inMNI_FULL_HISTO_MATCHED_VOLUME_LESIONI.nii.gz",
]

combined_mask_path = r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\sub08(check flair e t1)\volumes_generated_no_bandoni\output_overlay\combined_mask_GLIOMA_MS.nii.gz"
output_path       = r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\sub08(check flair e t1)\volumes_generated_no_bandoni\output_overlay\VOLUME_COMBINATO.nii.gz"


# ==========================================================
# 1) CARICO E NORMALIZZO 0–1 IL VOLUME DI RIFERIMENTO
# ==========================================================
print("Carico volume di riferimento...")
ref_img_raw = sitk.ReadImage(reference_path, sitk.sitkFloat32)
ref_arr = sitk.GetArrayFromImage(ref_img_raw).astype(np.float32)

ref_nonzero = ref_arr[ref_arr > 0]

if ref_nonzero.size > 0:
    r_min = ref_nonzero.min()
    r_max = ref_nonzero.max()

    ref_norm_arr = np.zeros_like(ref_arr, dtype=np.float32)

    if r_max > r_min:
        ref_norm_arr[ref_arr > 0] = (ref_arr[ref_arr > 0] - r_min) / (r_max - r_min)
    else:
        ref_norm_arr[ref_arr > 0] = 1.0
else:
    ref_norm_arr = np.zeros_like(ref_arr, dtype=np.float32)

ref_img = sitk.GetImageFromArray(ref_norm_arr)
ref_img.CopyInformation(ref_img_raw)

print("Volume di riferimento normalizzato 0–1.")


# ==========================================================
# 2) CARICO E NORMALIZZO 0–1 I VOLUMI LESIONALI + CREO MASCHERE
# ==========================================================
lesion_imgs = []
masks = []

for lesion_path in lesion_volumes_paths:
    print("Carico volume lesionale:", lesion_path)

    lesion_img_raw = sitk.ReadImage(lesion_path, sitk.sitkFloat32)
    lesion_arr = sitk.GetArrayFromImage(lesion_img_raw).astype(np.float32)

    lesion_nonzero = lesion_arr[lesion_arr > 0]

    if lesion_nonzero.size > 0:
        v_min = lesion_nonzero.min()
        v_max = lesion_nonzero.max()

        lesion_norm_arr = np.zeros_like(lesion_arr, dtype=np.float32)

        if v_max > v_min:
            lesion_norm_arr[lesion_arr > 0] = (lesion_arr[lesion_arr > 0] - v_min) / (v_max - v_min)
        else:
            lesion_norm_arr[lesion_arr > 0] = 1.0
    else:
        lesion_norm_arr = np.zeros_like(lesion_arr, dtype=np.float32)

    lesion_img = sitk.GetImageFromArray(lesion_norm_arr)
    lesion_img.CopyInformation(lesion_img_raw)

    lesion_imgs.append(lesion_img)

    mask = lesion_img > 0   # maschera binaria 0/1
    masks.append(mask)


# ==========================================================
# 3) COMBINO LE LESIONI NORMALIZZATE
# ==========================================================
combined_lesions = sitk.Image(ref_img.GetSize(), sitk.sitkFloat32)
combined_lesions.CopyInformation(ref_img)

combined_mask = sitk.Image(ref_img.GetSize(), sitk.sitkUInt8)
combined_mask.CopyInformation(ref_img)

for lesion_img, mask in zip(lesion_imgs, masks):

    combined_lesions = sitk.Mask(combined_lesions, sitk.Not(mask)) + sitk.Mask(lesion_img, mask)

    combined_mask = sitk.Or(combined_mask, sitk.Cast(mask, sitk.sitkUInt8))

sitk.WriteImage(combined_mask, combined_mask_path)
print("Maschera combinata salvata in:", combined_mask_path)


# ==========================================================
# 4) OVERLAY (FUSIONE) FINALE
# ==========================================================
combined_mask_f = sitk.Cast(combined_mask, sitk.sitkFloat32)

final_volume = ref_img * (1.0 - combined_mask_f) + combined_lesions * combined_mask_f

sitk.WriteImage(final_volume, output_path)
print("Overlay finale salvato in:", output_path)
