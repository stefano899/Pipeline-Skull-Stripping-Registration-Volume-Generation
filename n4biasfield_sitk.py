# -*- coding: utf-8 -*-
"""
Applica N4 bias field correction (via ANTsPy) a tutti i volumi NIfTI
presenti nelle cartelle 'skullstripped' (o nelle loro sotto-cartelle)
di tutti i soggetti 'sub*' in BASE_DIR.

Per ciascuna 'skullstripped', crea una cartella sorella
'volumi_coregistrati_alla_t1_bias' (ma in questo script la struttura
viene ricreata sotto OUTPUT_BASE_DIR) dove salva i volumi corretti.

Output filename: <subject>-<mod>-bias.nii.gz
"""

import sys
from pathlib import Path
import shutil
import ants  # libreria ANTsPy
import SimpleITK as sitk  # <-- aggiunto per usare N4 di SimpleITK
from tqdm import tqdm      # <-- aggiunto per le barre di avanzamento

# ================================
# PATH BASE E PARAMETRI
# ================================

BASE_DIR = Path(r"E:\Datasets\NIMH\ds005752-download_coregistrati_MNI")

# 🔹 NUOVA CARTELLA DI OUTPUT (RICHIESTA)
OUTPUT_BASE_DIR = Path(r"C:\Users\Stefano\Desktop\Stefano\Datasets\NihmBias")

# ================================
# OPZIONE DI SOVRASCRITTURA
# ================================
OVERWRITE_EXISTING = True   # True  = elimina e ricrea la cartella di output
                            # False = mantiene la cartella e salta i file già creati

# ================================
# FORMATI NIFTI
# ================================
NIFTI_EXTS = (".nii.gz", ".nii")


# ================================
# FUNZIONI DI SUPPORTO
# ================================

def subjects_under(base: Path):
    """Ritorna tutte le cartelle soggetto: directory che iniziano con 'sub'.

    🔹 MODIFICATO: gestisce directory illeggibili (WinError 1392, ecc.)
    """
    if not base.exists():
        raise FileNotFoundError(f"Base non trovata: {base}")
    for d in sorted(base.iterdir()):
        try:
            if d.is_dir() and d.name.lower().startswith("sub"):
                yield d
        except OSError as e:
            # Qui saltiamo la directory marcia invece di far crashare tutto
            print(f"[WARN] Salto directory illeggibile: {d} ({e})")


def is_nifti(path: Path) -> bool:
    """True se il file ha estensione NIfTI (.nii / .nii.gz)."""
    return any(path.name.lower().endswith(ext) for ext in NIFTI_EXTS)


def find_volcoreg_dirs(subject_dir: Path):
    """
    Cerca tutte le cartelle chiamate 'skullstripped' sotto il soggetto.
    Esempio:
        sub01/anat/skullstripped
    """
    for p in subject_dir.rglob("skullstripped"):
        try:
            if p.is_dir():
                yield p
        except OSError as e:
            print(f"[WARN] Salto cartella 'skullstripped' illeggibile: {p} ({e})")


def guess_subject_and_modality(path: Path, default_subj: str):
    """
    Estrae:
      - subject: da 'sub-XXXX' nel nome del file (se presente), altrimenti usa default_subj
      - modality: UNA tra 'T1', 'T2', 'FLAIR' cercando il pattern nel nome file (case-insensitive).
    """
    name = path.name
    base = name
    # rimuove estensione .nii.gz o .nii
    for ext in NIFTI_EXTS:
        if base.endswith(ext):
            base = base[:-len(ext)]
            break

    parts = base.split("_")

    # Subject dal filename (primo pezzo se inizia con "sub-")
    if parts and parts[0].startswith("sub-"):
        subj_id = parts[0]
    else:
        subj_id = default_subj

    # Estrazione modalità
    b_low = base.lower()
    if "flair" in b_low:
        modality = "FLAIR"
    elif "t2" in b_low:   # copre 't2', 't2w', ecc.
        modality = "T2"
    elif "t1" in b_low:   # copre 't1', 't1w', ecc.
        modality = "T1"
    else:
        modality = "unk"

    return subj_id, modality


def run_n4_bias(input_nii: Path, output_nii: Path):
    """Esegue N4 bias field correction con SimpleITK su un singolo volume."""
    print(f"      [N4/SITK] N4 su: {input_nii}")
    try:
        inputImage = sitk.ReadImage(str(input_nii), sitk.sitkFloat32)
        image = inputImage

        maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)

        shrinkFactor = 1
        if shrinkFactor > 1:
            image = sitk.Shrink(
                inputImage, [shrinkFactor] * inputImage.GetDimension()
            )
            maskImage = sitk.Shrink(
                maskImage, [shrinkFactor] * inputImage.GetDimension()
            )

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        numberFittingLevels = 4
        # Se vuoi, puoi impostare esplicitamente le iterazioni:
        # corrector.SetMaximumNumberOfIterations([50] * numberFittingLevels)

        corrected_image = corrector.Execute(image, maskImage)
        log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
        corrected_image_full_resolution = inputImage / sitk.Exp(log_bias_field)

        # Assicurati che la cartella di output esista
        output_nii.parent.mkdir(parents=True, exist_ok=True)

        sitk.WriteImage(corrected_image_full_resolution, str(output_nii))

        print(f"      ✅ Salvato: {output_nii}")
    except Exception as e:
        print(f"      ❌ ERRORE N4 su {input_nii}: {e}")


def collect_nifti_files(vol_dir: Path):
    """
    Raccoglie tutti i file NIfTI presenti in una cartella 'skullstripped'.

    - Prima cerca .nii/.nii.gz direttamente in vol_dir.
    - Se non ne trova, cerca ricorsivamente nelle sotto-cartelle.
    """
    print(f"      [DEBUG] Cerco NIfTI in: {vol_dir}")

    direct_files = []
    try:
        for f in sorted(vol_dir.iterdir()):
            try:
                if f.is_file() and is_nifti(f):
                    direct_files.append(f)
            except OSError as e:
                print(f"      [WARN] Salto file illeggibile: {f} ({e})")
    except OSError as e:
        print(f"      [WARN] Impossibile iterare {vol_dir}: {e}")
        return []

    if direct_files:
        print(f"      [DEBUG] Trovati {len(direct_files)} file nella cartella principale.")
        return direct_files

    nested = []
    try:
        for f in vol_dir.rglob("*"):
            try:
                if f.is_file() and is_nifti(f):
                    nested.append(f)
            except OSError as e:
                print(f"      [WARN] Salto file (rglob) illeggibile: {f} ({e})")
    except OSError as e:
        print(f"      [WARN] Impossibile fare rglob in {vol_dir}: {e}")
        return []

    if nested:
        print(f"      [DEBUG] Trovati {len(nested)} file nelle sotto-cartelle.")
        return nested

    print("      ⚠️  Nessun NIfTI trovato.")
    return []


# ================================
# PIPELINE PRINCIPALE
# ================================

def main():
    print("[INFO] Avvio N4 bias correction")
    print(f"[INFO] BASE_DIR           = {BASE_DIR}")
    print(f"[INFO] OUTPUT_BASE_DIR    = {OUTPUT_BASE_DIR}")
    print(f"[INFO] OVERWRITE_EXISTING = {OVERWRITE_EXISTING}")
    print("============================================================\n")

    # Creo la cartella base di output se non esiste
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    if "ants" not in sys.modules:
        print("❌ ERRORE: modulo 'ants' non disponibile (antspyx).")
        sys.exit(1)

    total_subj = 0
    total_dirs = 0
    total_vols = 0

    subj_list = list(subjects_under(BASE_DIR))
    for subj in tqdm(subj_list, desc="Soggetti", unit="subj"):
        total_subj += 1
        subj_name = subj.name

        print("\n============================================================")
        print(f"[SOGGETTO] {subj_name}")

        vol_dirs = list(find_volcoreg_dirs(subj))
        if not vol_dirs:
            print("  ⚠️  Nessuna cartella 'skullstripped' trovata per questo soggetto.")
            continue

        print(f"  ✓ Trovate {len(vol_dirs)} cartelle 'skullstripped'.")
        total_dirs += len(vol_dirs)

        for vol_dir in tqdm(vol_dirs, desc=f"{subj_name} - skullstripped", unit="dir", leave=False):

            # 🔹 NUOVA LOGICA DI OUTPUT:
            # ricreo la struttura relativa a partire da BASE_DIR dentro OUTPUT_BASE_DIR
            try:
                rel_parent = vol_dir.parent.relative_to(BASE_DIR)
            except ValueError:
                # In casi strani in cui vol_dir non è sotto BASE_DIR,
                # metto tutto sotto una cartella del soggetto.
                rel_parent = Path(subj_name)

            out_dir = OUTPUT_BASE_DIR / rel_parent / "volumi_coregistrati_alla_t1_bias"

            # Se devo sovrascrivere, elimino completamente la cartella di output
            if OVERWRITE_EXISTING and out_dir.exists():
                print(f"  [CLEAN] Rimuovo cartella di output esistente: {out_dir}")
                shutil.rmtree(out_dir)

            # (ri)creo la cartella
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  [DIR] Input : {vol_dir}")
            print(f"       Output: {out_dir}")

            vols = collect_nifti_files(vol_dir)
            if not vols:
                continue

            for v in tqdm(vols, desc=f"{subj_name} - volumi", unit="vol", leave=False):

                if "mask" in v.name.lower():
                    print(f"      [SKIP] Volume di maschera (no N4): {v.name}")
                    continue

                subj_id, mod = guess_subject_and_modality(v, default_subj=subj_name)
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
    print("✅ Completato")
    print(f"   Soggetti elaborati : {total_subj}")
    print(f"   Cartelle trovate   : {total_dirs}")
    print(f"   Volumi processati  : {total_vols}")


if __name__ == "__main__":
    main()
