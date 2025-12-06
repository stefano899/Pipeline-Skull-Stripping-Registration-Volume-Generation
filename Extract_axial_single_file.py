import os
from pathlib import Path

import nibabel as nib
import numpy as np
import imageio


# ==============================
# CONFIG
# ==============================
INPUT_NIFTI = r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\sub08_check_flair_t1\volumes_generated_no_bandoni\output_overlay\VOLUME_COMBINATO.nii.gz"   # 🔴 Modifica qui
OUTPUT_ROOT = r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\sub08_check_flair_t1\volumes_generated_no_bandoni\output_overlay\temporaneo\output_slices"     # 🔴 E qui

# shape target del volume (X, Y, Z)
TARGET_SHAPE = (192, 228, 192)

# threshold per filtrare slice troppo vuote
NZ_THRESHOLD = 400


# ==============================
# FUNZIONI DI BASE
# ==============================
def load_and_normalize(path, is_mask=False):
    """
    Carica il volume NIfTI e lo normalizza in [0,1] dopo z-score.
    Tutto a livello di volume 3D.
    """
    img = nib.load(path)
    data = img.get_fdata()

    print("Shape volume ORIGINALE:", data.shape)

    # niente valori negativi
    data[data < 0] = 0

    if is_mask:
        return data

    # z-score sul volume intero
    mean = np.mean(data)
    std = np.std(data)
    if std > 0:
        data = (data - mean) / std
    else:
        data = np.zeros_like(data)

    # normalizzazione [0,1]
    dmin, dmax = np.nanmin(data), np.nanmax(data)
    if dmax > dmin:
        data = (data - dmin) / (dmax - dmin)
    else:
        data = np.zeros_like(data)

    return data


def find_trim_indices(ref_vol: np.ndarray, eps: float = 0.0):
    """
    Trova UNA slice per ciascun asse (X, Y, Z) scegliendo quella con
    meno voxel non-zero.
    """
    nz_counts_x = np.count_nonzero(ref_vol > eps, axis=(1, 2))
    idx_x = int(np.argmin(nz_counts_x))

    nz_counts_y = np.count_nonzero(ref_vol > eps, axis=(0, 2))
    idx_y = int(np.argmin(nz_counts_y))

    nz_counts_z = np.count_nonzero(ref_vol > eps, axis=(0, 1))
    idx_z = int(np.argmin(nz_counts_z))

    print(f"Slice da rimuovere: X={idx_x}, Y={idx_y}, Z={idx_z}")
    return idx_x, idx_y, idx_z


def apply_trim_indices(vol: np.ndarray, idx_x: int, idx_y: int, idx_z: int):
    """
    Rimuove 1 slice lungo ciascun asse (X, Y, Z) alle posizioni specificate.
    """
    vol_out = vol
    vol_out = np.delete(vol_out, idx_x, axis=0)
    vol_out = np.delete(vol_out, idx_y, axis=1)
    vol_out = np.delete(vol_out, idx_z, axis=2)
    return vol_out


def pad_or_crop_to_shape(vol: np.ndarray, target_shape):
    """
    Porta il volume (3D) a target_shape (X, Y, Z) con crop/padding simmetrico.
    """
    assert vol.ndim == 3, "Il volume deve essere 3D"
    x, y, z = vol.shape
    tx, ty, tz = target_shape
    out = vol

    # Asse 0 (X)
    if x > tx:
        excess = x - tx
        cut_before = excess // 2
        cut_after = excess - cut_before
        out = out[cut_before:x - cut_after, :, :]
    elif x < tx:
        deficit = tx - x
        pad_before = deficit // 2
        pad_after = deficit - pad_before
        out = np.pad(out, ((pad_before, pad_after), (0, 0), (0, 0)),
                     mode="constant", constant_values=0)

    # Asse 1 (Y)
    x, y, z = out.shape
    if y > ty:
        excess = y - ty
        cut_before = excess // 2
        cut_after = excess - cut_before
        out = out[:, cut_before:y - cut_after, :]
    elif y < ty:
        deficit = ty - y
        pad_before = deficit // 2
        pad_after = deficit - pad_before
        out = np.pad(out, ((0, 0), (pad_before, pad_after), (0, 0)),
                     mode="constant", constant_values=0)

    # Asse 2 (Z)
    x, y, z = out.shape
    if z > tz:
        excess = z - tz
        cut_before = excess // 2
        cut_after = excess - cut_before
        out = out[:, :, cut_before:z - cut_after]
    elif z < tz:
        deficit = tz - z
        pad_before = deficit // 2
        pad_after = deficit - pad_before
        out = np.pad(out, ((0, 0), (0, 0), (pad_before, pad_after)),
                     mode="constant", constant_values=0)

    print(f"Shape dopo pad/crop a target {target_shape}: {out.shape}")
    return out


def compute_valid_slice_indices(ref_vol_trimmed: np.ndarray, nz_threshold: int):
    """
    Decide quali indici di slice tenere per ciascuna orientazione,
    in base al numero di voxel non-zero (>= nz_threshold).
    """
    valid = {}

    # Assiale: lungo Z
    axial_indices = []
    for i in range(ref_vol_trimmed.shape[2]):
        sl = ref_vol_trimmed[:, :, i]
        img_uint8 = (sl * 255).astype(np.uint8)
        if np.count_nonzero(img_uint8) >= nz_threshold:
            axial_indices.append(i)
    valid["axial"] = axial_indices

    # Coronale: lungo Y
    coronal_indices = []
    for i in range(ref_vol_trimmed.shape[1]):
        sl = ref_vol_trimmed[:, i, :]
        img_uint8 = (sl * 255).astype(np.uint8)
        if np.count_nonzero(img_uint8) >= nz_threshold:
            coronal_indices.append(i)
    valid["coronal"] = coronal_indices

    # Sagittale: lungo X
    sagittal_indices = []
    for i in range(ref_vol_trimmed.shape[0]):
        sl = ref_vol_trimmed[i, :, :]
        img_uint8 = (sl * 255).astype(np.uint8)
        if np.count_nonzero(img_uint8) >= nz_threshold:
            sagittal_indices.append(i)
    valid["sagittal"] = sagittal_indices

    print("Indici validi (ref trimmato + target shape):")
    print(f"  axial   : {len(valid['axial'])} slice")
    print(f"  coronal : {len(valid['coronal'])} slice")
    print(f"  sagittal: {len(valid['sagittal'])} slice")

    return valid


def save_slices_for_orientation(vol,
                                orientation: str,
                                output_dir: str,
                                slice_indices):
    """
    Estrae e salva le slice di una certa orientazione.
    Nome file: basenameORI_slice_XXX.png
    """
    os.makedirs(output_dir, exist_ok=True)

    if orientation == 'axial':
        n_slices = vol.shape[2]
        slicer = lambda idx: vol[:, :, idx]
    elif orientation == 'coronal':
        n_slices = vol.shape[1]
        slicer = lambda idx: vol[:, idx, :]
    elif orientation == 'sagittal':
        n_slices = vol.shape[0]
        slicer = lambda idx: vol[idx, :, :]
    else:
        raise ValueError("orientation deve essere 'axial', 'coronal' o 'sagittal'.")

    if slice_indices is None:
        indices = range(n_slices)
    else:
        indices = slice_indices

    count = 0
    for i in indices:
        sl = slicer(i)
        img_uint8 = (sl * 255).astype(np.uint8)

        fname = f"{orientation}_slice_{i:03d}.png"
        out_path = os.path.join(output_dir, fname)
        imageio.imwrite(out_path, img_uint8)
        count += 1

    print(f"Salvate {count} slice in orientazione {orientation} -> {output_dir}")


# ==============================
# PIPELINE PER UN SOLO FILE
# ==============================
def process_single_nifti(nifti_path: str,
                         output_root: str,
                         target_shape=(192, 228, 192),
                         nz_threshold=400):
    nifti_path = Path(nifti_path)
    if not nifti_path.is_file():
        raise FileNotFoundError(f"NIfTI non trovato: {nifti_path}")

    print(f"\n=== Processing: {nifti_path} ===")

    # cartella di output per questo volume
    base_name = nifti_path.stem
    if base_name.endswith(".nii"):
        base_name = base_name[:-4]  # rimuove eventuale .nii
    vol_out_root = Path(output_root) / base_name
    os.makedirs(vol_out_root, exist_ok=True)

    # 1) carica e normalizza
    vol = load_and_normalize(str(nifti_path), is_mask=False)

    # 2) trova indici di trim
    idx_x, idx_y, idx_z = find_trim_indices(vol, eps=0.0)

    # 3) applica trim
    vol = apply_trim_indices(vol, idx_x, idx_y, idx_z)
    print("Shape dopo trim:", vol.shape)

    # 4) pad/crop alla shape target
    vol = pad_or_crop_to_shape(vol, target_shape)

    # 5) calcola indici validi per ogni orientazione
    valid_slice_indices = compute_valid_slice_indices(vol, nz_threshold)

    # 6) salva slice per le tre orientazioni
    for ori in ["axial", "coronal", "sagittal"]:
        ori_dir = vol_out_root / ori
        save_slices_for_orientation(
            vol=vol,
            orientation=ori,
            output_dir=str(ori_dir),
            slice_indices=valid_slice_indices[ori],
        )

    print("\n✅ Completato.")


if __name__ == "__main__":
    process_single_nifti(
        nifti_path=INPUT_NIFTI,
        output_root=OUTPUT_ROOT,
        target_shape=TARGET_SHAPE,
        nz_threshold=NZ_THRESHOLD,
    )
