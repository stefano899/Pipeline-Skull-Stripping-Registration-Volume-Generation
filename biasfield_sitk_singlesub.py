# -*- coding: utf-8 -*-
"""
Versione per singolo soggetto: imposta TARGET_SUBJECT = "subXX"
"""

import sys
from pathlib import Path
import shutil
import ants
import SimpleITK as sitk
from tqdm import tqdm

# ================================
# PARAMETRI
# ================================

BASE_DIR = Path(r"E:\Datasets\Volumi_sani_T1_E_FLAIR_1mm_MNI")

TARGET_SUBJECT = "sub130"   # <<< QUI scegli il soggetto da processare!

OVERWRITE_EXISTING = True
NIFTI_EXTS = (".nii.gz", ".nii")


# ================================
# FUNZIONI DI SUPPORTO
# ================================

def subjects_under(base: Path):
    for d in sorted(base.iterdir()):
        if d.is_dir() and d.name.lower().startswith("sub"):
            yield d


def is_nifti(path: Path) -> bool:
    return any(path.name.lower().endswith(ext) for ext in NIFTI_EXTS)


def find_volcoreg_dirs(subject_dir: Path):
    for p in subject_dir.rglob("skullstripped"):
        if p.is_dir():
            yield p


def guess_subject_and_modality(path: Path, default_subj: str):
    name = path.name
    base = name

    for ext in NIFTI_EXTS:
        if base.endswith(ext):
            base = base[:-len(ext)]
            break

    parts = base.split("_")
    if parts and parts[0].startswith("sub-"):
        subj_id = parts[0]
    else:
        subj_id = default_subj

    b_low = base.lower()
    if "flair" in b_low:
        modality = "FLAIR"
    elif "t2" in b_low:
        modality = "T2"
    elif "t1" in b_low:
        modality = "T1"
    else:
        modality = "unk"

    return subj_id, modality


def run_n4_bias(input_nii: Path, output_nii: Path):
    print(f"      [N4/SITK] N4 su: {input_nii}")
    try:
        inputImage = sitk.ReadImage(str(input_nii), sitk.sitkFloat32)
        image = inputImage
        maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)

        shrinkFactor = 1

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        numberFittingLevels = 4

        corrected_image = corrector.Execute(image, maskImage)
        log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
        corrected_image_full_resolution = inputImage / sitk.Exp(log_bias_field)

        sitk.WriteImage(corrected_image_full_resolution, str(output_nii))

        print(f"      ✅ Salvato: {output_nii}")
    except Exception as e:
        print(f"      ❌ ERRORE N4 su {input_nii}: {e}")


def collect_nifti_files(vol_dir: Path):
    print(f"      [DEBUG] Cerco NIfTI in: {vol_dir}")

    direct_files = [f for f in sorted(vol_dir.iterdir()) if f.is_file() and is_nifti(f)]

    if direct_files:
        print(f"      [DEBUG] File trovati nella cartella principale: {len(direct_files)}")
        return direct_files

    nested = [f for f in vol_dir.rglob("*") if f.is_file() and is_nifti(f)]

    if nested:
        print(f"      [DEBUG] File trovati nelle sotto-cartelle: {len(nested)}")
        return nested

    print("      ⚠️ Nessun NIfTI trovato.")
    return []


# ================================
# PIPELINE PER SINGOLO SOGGETTO
# ================================

def main():
    print("[INFO] Avvio N4 bias correction (versione singolo soggetto)")
    print(f"[INFO] BASE_DIR        = {BASE_DIR}")
    print(f"[INFO] TARGET_SUBJECT  = {TARGET_SUBJECT}")
    print("============================================================\n")

    # --- cerca il soggetto richiesto ---
    subj_dir = BASE_DIR / TARGET_SUBJECT

    if not subj_dir.exists():
        print(f"❌ ERRORE: soggetto {TARGET_SUBJECT} non trovato in {BASE_DIR}")
        sys.exit(1)

    print(f"[SOGGETTO] {TARGET_SUBJECT}")

    vol_dirs = list(find_volcoreg_dirs(subj_dir))

    if not vol_dirs:
        print("  ⚠️ Nessuna cartella 'skullstripped' trovata per questo soggetto.")
        sys.exit(1)

    print(f"  ✓ Trovate {len(vol_dirs)} cartelle 'skullstripped'.")

    total_vols = 0

    # tqdm sulle cartelle skullstripped
    for vol_dir in tqdm(vol_dirs, desc="Cartelle skullstripped", unit="dir"):

        out_dir = vol_dir.parent / "volumi_coregistrati_alla_t1_bias"

        # pulizia cartella output
        if OVERWRITE_EXISTING and out_dir.exists():
            print(f"  [CLEAN] Rimuovo cartella: {out_dir}")
            shutil.rmtree(out_dir)

        out_dir.mkdir(exist_ok=True)

        print(f"\n  [DIR] Input : {vol_dir}")
        print(f"       Output: {out_dir}")

        vols = collect_nifti_files(vol_dir)
        if not vols:
            continue

        # tqdm sui volumi
        for v in tqdm(vols, desc="Volumi", unit="vol", leave=False):

            if "mask" in v.name.lower():
                print(f"      [SKIP] Volume di maschera: {v.name}")
                continue

            subj_id, mod = guess_subject_and_modality(v, default_subj=TARGET_SUBJECT)
            out_name = f"{subj_id}-{mod}-bias.nii.gz"
            out_path = out_dir / out_name

            if out_path.exists() and not OVERWRITE_EXISTING:
                print(f"      [SKIP] Esiste già: {out_path}")
                continue

            print(f"\n      [N4] Input : {v}")
            print(f"           Subject: {subj_id}")
            print(f"           Mod    : {mod}")
            print(f"           Out    : {out_path}")

            run_n4_bias(v, out_path)
            total_vols += 1

    print("\n============================================================")
    print(f"✅ Completato soggetto {TARGET_SUBJECT}")
    print(f"   Volumi processati: {total_vols}")


if __name__ == "__main__":
    main()
