import SimpleITK as sitk
import numpy as np


def histogram_matching_intensity_gt_zero(
    moving_path: str,
    reference_path: str,
    output_path: str,
):
    """
    Esegue histogram matching tra due volumi NIfTI usando solo i voxel con intensità > 0.

    - moving_path: NIfTI da normalizzare (verrà modificato)
    - reference_path: NIfTI di riferimento
    - output_path: dove salvare il risultato
    """

    # 1) Leggo le immagini come float32
    moving_img = sitk.ReadImage(moving_path, sitk.sitkFloat32)
    ref_img = sitk.ReadImage(reference_path, sitk.sitkFloat32)

    moving = sitk.GetArrayFromImage(moving_img)   # shape: [z, y, x]
    reference = sitk.GetArrayFromImage(ref_img)

    # 2) Mask: consideriamo solo intensità > 0
    moving_mask = moving > 0
    reference_mask = reference > 0

    moving_vals = moving[moving_mask].ravel()
    ref_vals = reference[reference_mask].ravel()

    if moving_vals.size == 0 or ref_vals.size == 0:
        raise ValueError("Una delle immagini non ha voxel con intensità > 0.")

    # 3) Histogram matching "a mano" con NumPy (stile skimage.exposure.match_histograms)
    #    a) valori unici e conteggi
    s_values, s_idx, s_counts = np.unique(
        moving_vals, return_inverse=True, return_counts=True
    )
    t_values, t_counts = np.unique(ref_vals, return_counts=True)

    #    b) CDF delle due distribuzioni
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]

    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    #    c) mappo i quantili della sorgente su quelli del template
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    #    d) ricostruisco i valori matched
    matched_vals = interp_t_values[s_idx]

    # 4) Rimetto i valori matched solo dove mask > 0, il resto rimane invariato
    matched = moving.copy()
    matched[moving_mask] = matched_vals

    # 5) Ricostruisco immagine SimpleITK preservando metadati
    matched_img = sitk.GetImageFromArray(matched)
    matched_img.CopyInformation(moving_img)

    sitk.WriteImage(matched_img, output_path)
    return matched_img


if __name__ == "__main__":
    moving_path = r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\testing_08_patient_07 (per lesioni)\Testing_Center_08_Patient_07_FLAIR_preprocessed_inMNI_FULL.nii.gz"
    reference_path = r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\BraTS-GLI-00000-000\Registered\BraTS-GLI-00000-000-t2f.nii_in_MNI_FULL.nii.gz"
    output_path = r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\testing_08_patient_07 (per lesioni)\Testing_Center_08_Patient_07_FLAIR_preprocessed_inMNI_FULL_HISTO_MATCHED.nii.gz"

    histogram_matching_intensity_gt_zero(moving_path, reference_path, output_path)
    print(f"Salvato: {output_path}")
