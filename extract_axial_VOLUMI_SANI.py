import os
import shutil
import nibabel as nib
import numpy as np
import imageio
from pathlib import Path
import random
from datetime import datetime

# proviamo a usare tqdm per la barra di avanzamento
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ========================================
# CONFIG GENERALE
# ========================================
base_root = r"E:\Datasets\Volumi_sani_T1_E_FLAIR_1mm_MNI"

# 🔹 output globale per il TRAIN
TRAIN_ROOT = r"C:\Users\Stefano\Desktop\Stefano\CycleGan\pytorch-CycleGAN-and-pix2pix\data_training\training_T1_FLAIR_Sani_Mni_Bias\train"
GLOBAL_TRAIN_A = os.path.join(TRAIN_ROOT, "trainA")  # es: T1
GLOBAL_TRAIN_B = os.path.join(TRAIN_ROOT, "trainB")  # es: FLAIR
TARGET_MODALITY_trainA = "T1"
TARGET_MODALITY_trainB = "FLAIR"

# 🔹 output strutturato per il TEST
TEST_ROOT = r"C:\Users\Stefano\Desktop\Stefano\CycleGan\pytorch-CycleGAN-and-pix2pix\data_training\training_T1_FLAIR_Sani_Mni_Bias\test"

os.makedirs(GLOBAL_TRAIN_A, exist_ok=True)
os.makedirs(GLOBAL_TRAIN_B, exist_ok=True)
os.makedirs(TEST_ROOT, exist_ok=True)

# 🔴 se False non sovrascrive i PNG esistenti (logica di base)
OVERWRITE = True

# 🔹 se True, NON cancella Output e NON sovrascrive niente: aggiunge solo i file mancanti
ADD_ONLY_MISSING = False

# variabile interna: cosa facciamo davvero sui file?
EFFECTIVE_OVERWRITE = OVERWRITE and not ADD_ONLY_MISSING

# modalità di riferimento per trovare le slice da togliere e da tenere
ref_modality = "FLAIR"

# threshold per tenere/filtrare slice troppo vuote (calcolato SOLO su ref_modality)
nz_threshold = 500

# 🔹 shape target del volume dopo trim: (X, Y, Z)
TARGET_SHAPE = (192, 228, 192)  # -> slice assiali 192x228

# ========================================
# CONFIG SPLIT TRAIN/TEST
# ========================================
TRAIN_FRACTION = 0.8
N_TRAIN_SUBJECTS = None  # se != None, ignora TRAIN_FRACTION e usa questo numero assoluto

# fissiamo seed per riproducibilità
random.seed(40)

# ========================================
# DISCOVERY DELLE CARTELLE
# ========================================
def discover_subject_anat_pairs(root: str):
    """
    Ritorna tutte le coppie (subj, path_relativo) dove esiste
    'anat/volumi_coregistrati_alla_t1_bias'.

    Se NON esiste quella cartella, in fallback usa 'anat/skullstripped'.
    Gestisce sia sub/... che sub/ses-.../...
    """
    root_p = Path(root)

    for subj in sorted(root_p.iterdir(), key=lambda p: p.name):
        if not subj.is_dir():
            continue
        if not subj.name.lower().startswith("sub"):
            continue

        # ===== Caso 1: subXX/anat/...
        anat_dir = subj / "anat"
        if anat_dir.is_dir():
            bias_dir = anat_dir / "volumi_coregistrati_alla_t1_bias"
            skull_dir = anat_dir / "skullstripped"

            if bias_dir.is_dir():
                # PRIORITÀ: cartella bias
                yield (subj.name, "anat/volumi_coregistrati_alla_t1_bias")
            elif skull_dir.is_dir():
                # fallback opzionale sui grezzi
                yield (subj.name, "anat/skullstripped")

        # ===== Caso 2: subXX/ses-YY/anat/...
        for ses in sorted(subj.glob("ses-*"), key=lambda p: p.name):
            ses_anat = ses / "anat"
            if not ses_anat.is_dir():
                continue

            bias_dir = ses_anat / "volumi_coregistrati_alla_t1_bias"
            skull_dir = ses_anat / "skullstripped"

            if bias_dir.is_dir():
                # PRIORITÀ: cartella bias nella sessione
                rel = bias_dir.relative_to(subj)
                yield (subj.name, str(rel))
            elif skull_dir.is_dir():
                # fallback opzionale sui grezzi
                rel = skull_dir.relative_to(subj)
                yield (subj.name, str(rel))

# ========================================
# PATH PER LE MODALITÀ (dentro skullstripped/bias)
# ========================================
def path_modality(subject: str, rel_patient: str, modality: str):
    folder = os.path.join(base_root, subject, rel_patient)
    if not os.path.isdir(folder):
        return None

    patterns = {
        "FLAIR": ["flair"],
        "T1": ["t1w", "mprage", "t1", "T1"],
        "T2": ["t2w", "_t2", "-t2", "t2.", "T2"],
        "PD": ["_pd", "pd.", "PD"],
        "T1c": ["t1c", "t1_gad", "t1-contr", "t1ce"],
        "GADO": ["gado"],
    }

    wanted = patterns.get(modality, [])

    for f in os.listdir(folder):
        f_low = f.lower()

        if not (f_low.endswith(".nii") or f_low.endswith(".nii.gz")):
            continue
        if "mask" in f_low:
            continue

        # se contiene sia flair che t2 -> lo considero flair, non T2
        if "flair" in f_low and "t2" in f_low:
            if modality == "FLAIR":
                return os.path.join(folder, f)
            if modality == "T2":
                continue

        if modality == "T2":
            if "flair" in f_low:
                continue
            if "t2star" in f_low or "t2*" in f_low:
                continue

        if modality == "FLAIR" and "flair" in f_low:
            return os.path.join(folder, f)

        if any(p in f_low for p in wanted):
            return os.path.join(folder, f)

    return None


# ========================================
# OUTPUT LOCALE
# ========================================
def output_root(subject: str, rel_patient: str) -> str:
    return os.path.join(base_root, subject, rel_patient, "Output", "ALL_AXES")


# ========================================
# FUNZIONI DI ELABORAZIONE
# ========================================
def load_and_normalize(path, is_mask=False):
    """
    Carica il volume così com'è nel NIfTI e lo normalizza in [0,1]
    dopo lo z-score (valori negativi tagliati a 0 all'inizio).
    Tutto fatto a livello di volume 3D.
    """
    img = nib.load(path)

    # Shape del volume originale
    print("  Shape volume ORIGINALE:", img.shape)

    data = img.get_fdata()

    # niente valori negativi prima di tutto
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

    # normalizzazione [0,1] sul volume intero
    dmin, dmax = np.nanmin(data), np.nanmax(data)
    if dmax > dmin:
        data = (data - dmin) / (dmax - dmin)
    else:
        data = np.zeros_like(data)

    return data


def find_trim_indices(ref_vol: np.ndarray, eps: float = 0.0):
    """
    Trova UNA slice per ciascun asse (X, Y, Z) nel volume di riferimento,
    scegliendo ogni volta la slice con meno voxel non-zero.
    Non modifica il volume, ritorna solo gli indici.
    """
    # Asse 0 (X)
    nz_counts_x = np.count_nonzero(ref_vol > eps, axis=(1, 2))
    idx_x = int(np.argmin(nz_counts_x))

    # Asse 1 (Y)
    nz_counts_y = np.count_nonzero(ref_vol > eps, axis=(0, 2))
    idx_y = int(np.argmin(nz_counts_y))

    # Asse 2 (Z)
    nz_counts_z = np.count_nonzero(ref_vol > eps, axis=(0, 1))
    idx_z = int(np.argmin(nz_counts_z))

    print(f"  Slice da rimuovere (ref_modality): X={idx_x}, Y={idx_y}, Z={idx_z}")
    return idx_x, idx_y, idx_z


def apply_trim_indices(vol: np.ndarray, idx_x: int, idx_y: int, idx_z: int):
    """
    Applica gli indici di trim a un volume:
    rimuove 1 slice lungo ciascun asse in posizione (idx_x, idx_y, idx_z).
    Operazione eseguita sul volume 3D.
    """
    vol_out = vol
    # ordine: prima X, poi Y, poi Z
    vol_out = np.delete(vol_out, idx_x, axis=0)
    vol_out = np.delete(vol_out, idx_y, axis=1)
    vol_out = np.delete(vol_out, idx_z, axis=2)
    return vol_out


def pad_or_crop_to_shape(vol: np.ndarray, target_shape):
    """
    Porta il volume (3D) a target_shape (X, Y, Z) tagliando dalle estremità
    o aggiungendo slice nere (padding) in modo simmetrico.
    Nessuna operazione è fatta sulle immagini 2D, solo sul volume.
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

    print(f"  Shape dopo pad/crop a target {target_shape}: {out.shape}")
    return out


def build_slice_name(subject, rel_patient, modality, slice_idx, ext=".png"):
    """
    Nome file = subject[_session]_MODALITY_slice_XXX.png
    Dove session è opzionale (solo se nel path c'è qualcosa tipo 'ses-01').
    """
    session = None
    parts = rel_patient.replace("\\", "/").split("/")
    for p in parts:
        if p.lower().startswith("ses-"):
            session = p
            break

    modality = modality.upper()

    if session:
        return f"{subject}_{session}_{modality}_slice_{slice_idx:03d}{ext}"
    else:
        return f"{subject}_{modality}_slice_{slice_idx:03d}{ext}"


def save_png_respecting_overwrite(img_array_uint8, dst_folder, filename):
    os.makedirs(dst_folder, exist_ok=True)
    out_path = os.path.join(dst_folder, filename)

    if EFFECTIVE_OVERWRITE:
        imageio.imwrite(out_path, img_array_uint8)
        return

    if not os.path.exists(out_path):
        imageio.imwrite(out_path, img_array_uint8)
        return
    # altrimenti non sovrascrivo
    return


def compute_valid_slice_indices(ref_vol_trimmed: np.ndarray, nz_threshold: int):
    """
    Usa SOLO il volume di riferimento già trimmato e portato alla shape target
    per decidere quali indici di slice tenere per ciascuna orientazione.
    Ritorna un dict:
        {
          "axial":    [idx1, idx2, ...],
          "coronal":  [idx1, idx2, ...],
          "sagittal": [idx1, idx2, ...]
        }
    """
    valid = {}

    # Assiale: slice lungo Z
    axial_indices = []
    for i in range(ref_vol_trimmed.shape[2]):
        sl = ref_vol_trimmed[:, :, i]
        img_uint8 = (sl * 255).astype(np.uint8)
        if np.count_nonzero(img_uint8) >= nz_threshold:
            axial_indices.append(i)
    valid["axial"] = axial_indices

    # Coronale: slice lungo Y
    coronal_indices = []
    for i in range(ref_vol_trimmed.shape[1]):
        sl = ref_vol_trimmed[:, i, :]
        img_uint8 = (sl * 255).astype(np.uint8)
        if np.count_nonzero(img_uint8) >= nz_threshold:
            coronal_indices.append(i)
    valid["coronal"] = coronal_indices

    # Sagittale: slice lungo X
    sagittal_indices = []
    for i in range(ref_vol_trimmed.shape[0]):
        sl = ref_vol_trimmed[i, :, :]
        img_uint8 = (sl * 255).astype(np.uint8)
        if np.count_nonzero(img_uint8) >= nz_threshold:
            sagittal_indices.append(i)
    valid["sagittal"] = sagittal_indices

    print("  Indici validi (calcolati su ref_modality trimmato + target shape):")
    print(f"    axial   : {len(valid['axial'])} slice")
    print(f"    coronal : {len(valid['coronal'])} slice")
    print(f"    sagittal: {len(valid['sagittal'])} slice")

    return valid


def save_slices_for_orientation(subject,
                                rel_patient,
                                modality,
                                vol,
                                orientation,
                                local_out_dir,
                                subject_in_test: bool,
                                slice_indices=None):
    """
    Estrae le slice lungo una certa orientazione.
    Nessun crop/zoom sulla singola immagine: la shape viene da TARGET_SHAPE.
    """
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

    os.makedirs(local_out_dir, exist_ok=True)

    if slice_indices is None:
        indices = range(n_slices)
    else:
        indices = slice_indices

    # recupero sessione (se esiste) per la struttura di test
    session = None
    parts = rel_patient.replace("\\", "/").split("/")
    for p in parts:
        if p.lower().startswith("ses-"):
            session = p
            break

    count = 0
    for i in indices:
        sl = slicer(i)          # slice 2D [0,1] già della dimensione vol[x,y]
        img_uint8 = (sl * 255).astype(np.uint8)

        fname = build_slice_name(subject, rel_patient, modality, i, ext=".png")

        # 1) salvataggio locale
        local_path = os.path.join(local_out_dir, fname)
        if EFFECTIVE_OVERWRITE or not os.path.exists(local_path):
            imageio.imwrite(local_path, img_uint8)

        # 2) salvataggio globale solo per AXIAL
        if orientation == "axial":
            if subject_in_test:
                # struttura per il test
                subj_root = os.path.join(TEST_ROOT, subject)
                if session:
                    base = os.path.join(subj_root, session, "test")
                else:
                    base = os.path.join(subj_root, "test")

                if modality.upper() == TARGET_MODALITY_trainA:
                    final_dir = os.path.join(base, "testA")
                elif modality.upper() == TARGET_MODALITY_trainB:
                    final_dir = os.path.join(base, "testB")
                else:
                    final_dir = None
            else:
                # struttura per il train
                if modality.upper() == TARGET_MODALITY_trainA:
                    final_dir = GLOBAL_TRAIN_A
                elif modality.upper() == TARGET_MODALITY_trainB:
                    final_dir = GLOBAL_TRAIN_B
                else:
                    final_dir = None

            if final_dir is not None:
                save_png_respecting_overwrite(img_array_uint8=img_uint8,
                                              dst_folder=final_dir,
                                              filename=fname)

        count += 1

    return count


# ========================================
# COPIA VOLUMI ORIGINALI NELLA STRUTTURA DI TEST
# ========================================
def copy_originals_to_test(subject, rel_patient, modalities_paths, subject_in_test: bool):
    """
    Se il soggetto è nel TEST, copia i volumi originali nella struttura:
      TEST_ROOT/subject[/ses-XX]/anat/skullstripped/
    senza sovrascrivere se EFFECTIVE_OVERWRITE = False.
    """
    if not subject_in_test:
        return

    session = None
    parts = rel_patient.replace("\\", "/").split("/")
    for p in parts:
        if p.lower().startswith("ses-"):
            session = p
            break

    subj_root = os.path.join(TEST_ROOT, subject)
    if session:
        skull_dir = os.path.join(subj_root, session, "anat", "skullstripped")
    else:
        skull_dir = os.path.join(subj_root, "anat", "skullstripped")

    os.makedirs(skull_dir, exist_ok=True)

    for mod, src_path in modalities_paths.items():
        if src_path is None or not os.path.isfile(src_path):
            continue
        fname = os.path.basename(src_path)
        dst_path = os.path.join(skull_dir, fname)

        if os.path.exists(dst_path) and not EFFECTIVE_OVERWRITE:
            continue

        shutil.copy2(src_path, dst_path)
        print(f"   [TEST COPY] {mod}: {src_path} -> {dst_path}")


# ========================================
# PROCESS DI UN SOGGETTO/ANAT
# ========================================
def process_subject(subject: str, rel_patient: str, subject_in_test: bool):
    print(f"\n=== {subject} | {rel_patient} ===")

    skull_dir = os.path.join(base_root, subject, rel_patient)
    if not os.path.isdir(skull_dir):
        print(f"[SKIP] Nessuna cartella skullstripped qui: {skull_dir}")
        return False

    # output locale
    out_root = output_root(subject, rel_patient)
    out_parent = os.path.dirname(out_root)

    # pulizia solo se veramente in modalità overwrite piena
    if EFFECTIVE_OVERWRITE:
        if os.path.isdir(out_parent):
            print(f"   🧹 Output esistente trovato ({out_parent}), lo elimino e ricreo...")
            shutil.rmtree(out_parent)

    os.makedirs(out_root, exist_ok=True)

    # cerca volumi
    modalities_paths = {}
    for m in ["FLAIR", "T1", "T2", "PD", "T1c", "GADO"]:
        p = path_modality(subject, rel_patient, m)
        if p is not None and os.path.isfile(p):
            modalities_paths[m] = p

    if not modalities_paths:
        print(f"[SKIP] Nessuna modalità trovata in {skull_dir}")
        return False

    # copia i volumi originali nella struttura di TEST (se soggetto di test)
    copy_originals_to_test(subject, rel_patient, modalities_paths, subject_in_test)

    # volume di riferimento (FLAIR se presente, altrimenti T1/T2)
    if ref_modality in modalities_paths:
        ref_path = modalities_paths[ref_modality]
    else:
        ref_path = modalities_paths.get("T1") or modalities_paths.get("T2")
        if ref_path is None:
            print(f"[SKIP] Nessuna modalità di riferimento trovata.")
            return False

    print(f"  Volume di riferimento per trim e selezione slice: {ref_path}")
    ref_vol = load_and_normalize(ref_path, is_mask=False)

    # trova indici delle slice da rimuovere su ref_modality
    idx_x, idx_y, idx_z = find_trim_indices(ref_vol, eps=0.0)

    # volume di riferimento trimmato
    ref_trimmed = apply_trim_indices(ref_vol, idx_x, idx_y, idx_z)
    print(f"  Shape {ref_modality} dopo trim: {ref_trimmed.shape}")

    # porta il volume di riferimento alla shape target (192, 228, 192)
    ref_trimmed = pad_or_crop_to_shape(ref_trimmed, TARGET_SHAPE)

    # calcola gli indici di slice validi sul ref_vol trimmato + target shape
    valid_slice_indices = compute_valid_slice_indices(ref_trimmed, nz_threshold)

    orientations = ["axial", "coronal", "sagittal"]

    # per ogni modalità
    for mod, vol_path in modalities_paths.items():
        print(f"  -> Modalità: {mod} | file: {vol_path}")
        mod_dir = os.path.join(out_root, mod)
        os.makedirs(mod_dir, exist_ok=True)

        vol = load_and_normalize(vol_path, is_mask=False)
        print(f"    Shape {mod} prima del trim: {vol.shape}")

        # stesso trim del ref
        vol = apply_trim_indices(vol, idx_x, idx_y, idx_z)
        print(f"    Shape {mod} dopo trim: {vol.shape}")

        # stessa operazione di pad/crop a livello di volume
        vol = pad_or_crop_to_shape(vol, TARGET_SHAPE)

        total_saved = 0
        for ori in orientations:
            ori_dir = os.path.join(mod_dir, ori)
            os.makedirs(ori_dir, exist_ok=True)

            saved = save_slices_for_orientation(
                subject=subject,
                rel_patient=rel_patient,
                modality=mod,
                vol=vol,
                orientation=ori,
                local_out_dir=ori_dir,
                subject_in_test=subject_in_test,
                slice_indices=valid_slice_indices[ori],
            )
            total_saved += saved
            print(f"    Saved {saved} {mod} {ori} slices -> {ori_dir}")

        print(f"  => {mod}: {total_saved} slice totali salvate.")

    return True


# ========================================
# MAIN
# ========================================
if __name__ == "__main__":
    # 1) scopriamo tutte le coppie soggetto/rel
    pairs = list(discover_subject_anat_pairs(base_root))

    # soggetti unici che hanno almeno una coppia
    subjects_with_pairs = sorted({subj for subj, _ in pairs})
    n_total = len(subjects_with_pairs)

    # decidi quanti soggetti vanno in train
    if N_TRAIN_SUBJECTS is not None:
        n_train = max(1, min(N_TRAIN_SUBJECTS, n_total - 1))
    else:
        n_train = int(round(n_total * TRAIN_FRACTION))
        n_train = max(1, min(n_train, n_total - 1))

    n_test = n_total - n_train

    random.shuffle(subjects_with_pairs)
    train_subjects = set(subjects_with_pairs[:n_train])
    test_subjects = set(subjects_with_pairs[n_train:])

    print("==================================================")
    print(f"RUN estrazione: {datetime.now().isoformat(sep=' ', timespec='seconds')}")
    print(f"Soggetti totali con skullstripped/bias: {n_total}")
    print(f"In TRAIN: {len(train_subjects)}")
    print(f"In TEST : {len(test_subjects)}")
    print(f"MODALITÀ FILE: OVERWRITE={OVERWRITE} | ADD_ONLY_MISSING={ADD_ONLY_MISSING} | EFFECTIVE_OVERWRITE={EFFECTIVE_OVERWRITE}")
    print("==================================================")

    missing_skull = []

    total_pairs = len(pairs)
    pair_iter = pairs
    if tqdm is not None:
        pair_iter = tqdm(pairs, desc="Estrazione slice", unit="cartella")

    # 2) processiamo tutte le coppie, sapendo se il soggetto è nel test
    for idx, (subject, rel_patient) in enumerate(pair_iter, start=1):
        if tqdm is None:
            print(f"\n[{idx}/{total_pairs}] Processing {subject} | {rel_patient}")

        ok = process_subject(subject, rel_patient, subject in test_subjects)
        if not ok:
            missing_skull.append(f"{subject}/{rel_patient}")

    print("\n✅ Estrazione completata.")
    if missing_skull:
        print("\n⚠️ Soggetti/ANAT senza volumi utili:")
        for s in missing_skull:
            print(" -", s)
    else:
        print("Tutti avevano almeno una modalità utile.")

    # report finale solo sulle cartelle di train
    def count_png(folder):
        return sum(1 for f in Path(folder).glob("*.png"))

    count_A = count_png(GLOBAL_TRAIN_A)
    count_B = count_png(GLOBAL_TRAIN_B)

    print("\n📊 Riepilogo finale (solo train):")
    print(f"  - trainA ({TARGET_MODALITY_trainA}) : {count_A:,} immagini")
    print(f"  - trainB ({TARGET_MODALITY_trainB}) : {count_B:,} immagini")
    if count_A == count_B:
        print("✅ Le due cartelle hanno lo stesso numero di immagini (ottimo).")
    else:
        diff = abs(count_A - count_B)
        print(f"⚠️ Differenza di {diff} immagini tra trainA e trainB.")
