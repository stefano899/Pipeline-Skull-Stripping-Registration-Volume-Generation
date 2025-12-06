import os
import re
import sys
import time
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output
from PIL import Image

# ================================
# CONFIG PRINCIPALE (SINGOLA CARTELLA)
# ================================

# 🔹 Cartella che contiene le slice FLAIR in PNG da dare in input al modello
IMAGES_DATAROOT = Path(
    r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\sub08_check_flair_t1\volumes_generated_no_bandoni\output_overlay\temporaneo\output_slices\VOLUME_COMBINATO\axial"
)

# 🔹 Cartella dove salvare TUTTO l'output per questa esecuzione
WORK_ROOT = Path(
    r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\sub08_check_flair_t1\volumes_generated_no_bandoni\output_overlay\temporaneo\output_slices\VOLUME_COMBINATO\output"
)

# 🔹 Percorso a test.py del repo pytorch-CycleGAN-and-pix2pix
TEST_PY_PATH = Path(
    r"C:\Users\Stefano\Desktop\Stefano\CycleGan\pytorch-CycleGAN-and-pix2pix\test.py"
)

# 🔹 Radice dei checkpoints (cartella "checkpoints")
CHECKPOINTS_BASE = Path(
    r"C:\Users\Stefano\Desktop\Stefano\CycleGan\pytorch-CycleGAN-and-pix2pix\checkpoints"
)

# 🔹 Nome del modello/checkpoint da usare (fisso come richiesto)
FRIENDLY_NAME = "from_flair_to_t1"
CKPT_SUBDIR = "cycle_T1_FLAIR_SANI"  # sottocartella dentro checkpoints

# ================================
# OPZIONI GENERALI
# ================================
OVERWRITE_EXISTING = True   # True = rigenera/sovrascrive tutto
POLL_SECONDS = 5
MAX_WAIT_SECONDS = 60        # tempo massimo per attendere cartelle/file
REQUIRE_STABLE_IMAGES = True
REQUIRE_STABLE_FILE = True

# Argomenti fissi di test.py (uguali al codice originale)
FIXED_ARGS = [
    "--model", "test",
    "--no_dropout",
    "--num_test", "500",
    "--preprocess", "none",
    "--no_flip",
    "--input_nc", "1",
    "--output_nc", "1",
]

# Parametri ricostruzione volumi
ISO_ZOOMS = (1, 1, 1)
ORDER = 1
ASCENDING = True

# Parametri histogram matching
AXIS = 2
REF_INDEX = None  # None = slice centrale


# ================================
# FUNZIONI DI UTILITÀ
# ================================
def has_images(src: Path) -> bool:
    if not src.exists() or not src.is_dir():
        return False
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        if any(src.glob(ext)):
            return True
    return False


def count_images(src: Path) -> int:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    return sum(1 for ext in exts for _ in src.glob(ext)) if src.exists() else 0


def wait_until_exists(folder: Path, poll_s: int = POLL_SECONDS,
                      max_wait: Optional[int] = MAX_WAIT_SECONDS) -> bool:
    waited = 0
    while not folder.exists():
        print(f"   ⏳ In attesa che compaia: {folder} (tra {poll_s}s)")
        time.sleep(poll_s)
        if max_wait is not None:
            waited += poll_s
            if waited >= max_wait:
                print(f"   [TIMEOUT] Cartella non trovata dopo {waited}s: {folder}")
                return False
    return True


def wait_until_images_exist(
    folder: Path,
    poll_s: int = POLL_SECONDS,
    max_wait: Optional[int] = MAX_WAIT_SECONDS,
    require_stable: bool = REQUIRE_STABLE_IMAGES,
) -> bool:
    waited = 0
    last_count = -1
    stable_hits = 0
    while True:
        if folder.exists() and folder.is_dir():
            c = count_images(folder)
            if c > 0:
                if not require_stable:
                    return True
                if c == last_count:
                    stable_hits += 1
                else:
                    stable_hits = 0
                last_count = c
                if stable_hits >= 1:
                    return True
            else:
                print(f"   ⏳ Cartella presente ma ancora vuota: {folder}")
        else:
            print(f"   ⏳ Cartella non pronta: {folder}")

        time.sleep(poll_s)
        if max_wait is not None:
            waited += poll_s
            if waited >= max_wait:
                print(f"   [TIMEOUT] Immagini non trovate entro {waited}s in: {folder}")
                return False


def wait_until_file_ready(path: Path, poll_s: int = POLL_SECONDS,
                          max_wait: Optional[int] = MAX_WAIT_SECONDS,
                          require_stable: bool = REQUIRE_STABLE_FILE) -> bool:
    waited = 0
    last_size = -1
    stable_count = 0
    while True:
        if path.exists():
            if not require_stable:
                return True
            size = path.stat().st_size
            if size == last_size and size > 0:
                stable_count += 1
            else:
                stable_count = 0
            last_size = size
            if stable_count >= 1:
                return True
        else:
            print(f"   ⏳ In attesa del file: {path}")

        time.sleep(poll_s)
        if max_wait is not None:
            waited += poll_s
            if waited >= max_wait:
                print(f"   [TIMEOUT] File non pronto dopo {waited}s: {path}")
                return False


def safe_rmtree(path: Path):
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def safe_unlink(path: Path):
    try:
        if path.exists():
            path.unlink()
    except PermissionError:
        pass


# ================================
# 1) RUN TEST.PY SU UNA SOLA CARTELLA DI PNG
# ================================
def run_single_test(images_dataroot: Path, work_root: Path) -> Path:
    """
    Lancia test.py usando:
      - dataroot = images_dataroot (cartella con PNG FLAIR)
      - checkpoint from_flair_to_t1
    Ritorna la cartella test_latest corrispondente.
    """
    if not images_dataroot.exists() or not images_dataroot.is_dir():
        raise FileNotFoundError(f"Cartella PNG inesistente: {images_dataroot}")

    if not has_images(images_dataroot):
        raise RuntimeError(f"Nessuna immagine trovata in: {images_dataroot}")

    results_dir = work_root / "slices_generated"
    checkpoints_dir = CHECKPOINTS_BASE / CKPT_SUBDIR
    exp_dir = checkpoints_dir / FRIENDLY_NAME
    g_path = exp_dir / "latest_net_G.pth"

    test_root = results_dir / FRIENDLY_NAME / "test_latest"
    images_dir = test_root / "images"

    # Pulizia / overwrite
    if OVERWRITE_EXISTING:
        safe_rmtree(test_root)
    else:
        if has_images(images_dir):
            print(f"[SKIP TEST] Immagini già presenti in {images_dir}")
            return test_root

    if not g_path.exists():
        raise FileNotFoundError(f"Checkpoint mancante: {g_path}")

    cmd = [
        sys.executable,
        str(TEST_PY_PATH),
        "--dataroot", str(images_dataroot),
        "--name", FRIENDLY_NAME,
        *FIXED_ARGS,
        "--results_dir", str(results_dir),
        "--checkpoints_dir", str(checkpoints_dir),
    ]

    print("\n==============================")
    print(f"MODELLO : {FRIENDLY_NAME}")
    print(f"DATAROOT: {images_dataroot}")
    print(f"RES_DIR : {results_dir}")
    print("==============================\n")

    subprocess.run(cmd, check=True)

    # attendo che la cartella con le immagini appaia
    ok = wait_until_images_exist(images_dir)
    if not ok:
        raise RuntimeError(f"Le immagini di output non sono comparse in {images_dir}")

    print(f"✅ Test completato. Output in: {test_root}")
    return test_root


# ================================
# 2) ORGANIZZA IMMAGINI (sposta da images/ a cartella padre)
# ================================
LABEL_RE = re.compile(r"slice_\d+_(.+?)\.(?:png|jpg|jpeg|tif|tiff)$", re.IGNORECASE)


def normalize_label(label: str) -> str:
    return "".join(label.split("_"))


def extract_label_from_name(filename: str):
    m = LABEL_RE.search(filename)
    return normalize_label(m.group(1)) if m else None


def organize_images(src: Path, dst: Path, copy: bool = False) -> int:
    if not src.exists() or not src.is_dir():
        print(f"[SKIP] Sorgente inesistente: {src}")
        return 0

    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    files = [f for ext in exts for f in src.glob(ext)]
    if not files:
        print(f"[SKIP] Nessuna immagine trovata in: {src}")
        return 0

    dst.mkdir(parents=True, exist_ok=True)
    moved = 0
    for f in files:
        label = extract_label_from_name(f.name)
        if not label:
            print(f"[SKIP] Nome non conforme (manca 'slice_'): {f.name}")
            continue

        target_dir = dst / label
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f.name

        if target_path.exists():
            if OVERWRITE_EXISTING:
                safe_unlink(target_path)
            else:
                print(f"[SKIP] Esiste già: {target_path}")
                continue

        if copy:
            shutil.copy2(f, target_path)
            print(f"[COPIO] {f.name} -> {label}/")
        else:
            shutil.move(str(f), str(target_path))
            print(f"[SPOSTO] {f.name} -> {label}/")
        moved += 1

    print(f"[OK] Organizzati {moved} file in {dst}")
    return moved


# ================================
# 3) RICOSTRUISCI VOLUME DA SLICE FAKE
# ================================
def stack_slices_to_nifti(folder, out_path, ascending=True,
                          iso_zooms=(1, 1, 1), order=1):
    if os.path.isdir(out_path):
        out_path = os.path.join(out_path, "volume.nii.gz")

    allowed_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    rx_index = re.compile(r"_slice_(\d+)(?=[^0-9]|$)", flags=re.IGNORECASE)

    files = []
    for f in os.listdir(folder):
        p = os.path.join(folder, f)
        if not os.path.isfile(p):
            continue
        _, ext = os.path.splitext(f)
        if ext.lower() not in allowed_ext:
            continue
        m = rx_index.search(f)
        if m:
            idx = int(m.group(1))
            files.append((idx, p))

    if not files:
        print(f"[SKIP] Nessuna immagine trovata in {folder} con pattern '_slice_<N>'.")
        return None

    files.sort(key=lambda x: x[0], reverse=not ascending)

    first = np.asarray(Image.open(files[0][1]))
    if first.ndim == 3:
        first = first[..., 0]
    if first.ndim != 2:
        print(f"[SKIP] La prima slice non è 2D: shape={first.shape}")
        return None

    H, W = first.shape
    Z = len(files)
    vol = np.empty((H, W, Z), dtype=first.dtype)
    vol[..., 0] = first

    for k, (_, path) in enumerate(files[1:], start=1):
        arr = np.asarray(Image.open(path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.shape != (H, W):
            raise ValueError(
                f"Shape diversa in {os.path.basename(path)}: {arr.shape} vs {(H, W)}"
            )
        vol[..., k] = arr

    img_raw = nib.Nifti1Image(vol, np.eye(4))
    img_raw.header.set_xyzt_units('mm')

    resampled = resample_to_output(img_raw, voxel_sizes=iso_zooms, order=order)
    data32 = resampled.get_fdata(dtype=np.float32)
    resampled = nib.Nifti1Image(data32, resampled.affine, header=resampled.header)
    resampled.set_qform(resampled.affine, code=1)
    resampled.set_sform(resampled.affine, code=1)

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nib.save(resampled, out_path)

    affine_path = out_path.replace(".nii.gz", "_affine.txt")
    np.savetxt(affine_path, resampled.affine, fmt="%.6f")

    print(f"✅ Volume salvato: {out_path}")
    print(f"✅ Affine salvata: {affine_path}")
    return out_path


# ================================
# 4) HISTOGRAM MATCHING SUL VOLUME
# ================================
def load_nifti(nifti_path: str):
    img = nib.load(nifti_path)
    img_c = nib.as_closest_canonical(img)
    vol = img_c.get_fdata(dtype=np.float32)
    return vol, img_c.affine, img_c.header


def masked_histogram_match(src2d: np.ndarray, ref2d: np.ndarray) -> np.ndarray:
    src2d = src2d.astype(np.float32, copy=False)
    ref2d = ref2d.astype(np.float32, copy=False)
    out = src2d.copy()

    m_src = src2d > 0
    m_ref = ref2d > 0
    if not np.any(m_src) or not np.any(m_ref):
        return out

    s_vals = src2d[m_src]
    r_vals = ref2d[m_ref]
    q = np.linspace(0, 100, 1024, dtype=np.float32)
    s_q = np.percentile(s_vals, q)
    r_q = np.percentile(r_vals, q)

    eps = np.finfo(np.float32).eps
    s_q = np.maximum.accumulate(
        s_q + np.linspace(0, eps * 10, s_q.size, dtype=np.float32)
    )

    out_vals = np.interp(s_vals, s_q, r_q).astype(np.float32)
    out[m_src] = out_vals
    return out


def histogram_match_volume(nifti_path: str, out_path: str,
                           axis: int = 2, ref_index: Optional[int] = None):
    vol, affine, hdr = load_nifti(nifti_path)

    if axis not in (0, 1, 2):
        print(f"[WARN] axis={axis} non valido; uso axis=2")
        axis = 2

    moved = False
    if axis != 2:
        vol = np.moveaxis(vol, axis, 2)
        moved = True

    z = vol.shape[2]
    if ref_index is None:
        ref_index = z // 2
    else:
        ref_index = max(0, min(ref_index, z - 1))

    ref = vol[:, :, ref_index].astype(np.float32)
    out_vol = np.empty_like(vol, dtype=np.float32)
    for k in range(z):
        sl = vol[:, :, k].astype(np.float32)
        out_vol[:, :, k] = masked_histogram_match(sl, ref)

    if moved:
        out_vol = np.moveaxis(out_vol, 2, axis)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    nib.save(nib.Nifti1Image(out_vol, affine, hdr), out_path)
    print(f"✅ Volume matched salvato in:\n{out_path}")


# ================================
# MAIN
# ================================
def main():
    print("[INFO] Pipeline singola cartella | OVERWRITE_EXISTING =", OVERWRITE_EXISTING)
    print(f"[INFO] Cartella PNG input : {IMAGES_DATAROOT}")
    print(f"[INFO] Cartella di lavoro : {WORK_ROOT}")

    WORK_ROOT.mkdir(parents=True, exist_ok=True)

    # 1) RUN TEST.PY
    test_root = run_single_test(IMAGES_DATAROOT, WORK_ROOT)

    # 2) ORGANIZZA IMMAGINI (sposta da test_latest/images)
    src_images = test_root / "images"
    dst_root = test_root   # come nello script originale
    print(f"\n[ORGANIZZA] SRC = {src_images}")
    print(f"[ORGANIZZA] DST = {dst_root}")
    organize_images(src_images, dst_root, copy=False)

    # 3) RICOSTRUISCI VOLUME DALLE SLICE FAKE
    fake_folder = test_root / "fake"
    out_volume = WORK_ROOT / "volumes_generated" / "volume_fake_t1.nii.gz"

    if out_volume.exists() and OVERWRITE_EXISTING:
        safe_unlink(out_volume)
        safe_unlink(Path(str(out_volume).replace(".nii.gz", "_affine.txt")))

    if not fake_folder.exists():
        ok = wait_until_exists(fake_folder)
        if not ok:
            print("[ERRORE] Cartella fake non trovata, impossibile creare il volume.")
            return

    ready = wait_until_images_exist(fake_folder)
    if not ready:
        print("[ERRORE] Immagini fake non pronte, impossibile creare il volume.")
        return

    stack_slices_to_nifti(
        folder=str(fake_folder),
        out_path=str(out_volume),
        ascending=ASCENDING,
        iso_zooms=ISO_ZOOMS,
        order=ORDER,
    )

    # 4) HISTOGRAM MATCHING SUL VOLUME (opzionale ma incluso)
    matched_out = WORK_ROOT / "volumes_generated_no_bandoni" / "volume_fake_t1_matched.nii.gz"
    if matched_out.exists() and OVERWRITE_EXISTING:
        safe_unlink(matched_out)

    if wait_until_file_ready(out_volume):
        histogram_match_volume(
            nifti_path=str(out_volume),
            out_path=str(matched_out),
            axis=AXIS,
            ref_index=REF_INDEX,
        )
    else:
        print("[WARN] Volume fake non pronto, salto histogram matching.")

    print("\n✔️ Pipeline singola cartella completata.")


if __name__ == "__main__":
    main()
