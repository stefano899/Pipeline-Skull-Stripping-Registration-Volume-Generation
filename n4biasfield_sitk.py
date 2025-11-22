# -*- coding: utf-8 -*-
"""
Applica N4 bias field correction (via ANTsPy) a tutti i volumi NIfTI
presenti nelle cartelle 'skullstripped' (o nelle loro sotto-cartelle)
di tutti i soggetti 'sub*' in BASE_DIR.

Per ciascuna 'skullstripped', crea una cartella sorella
'volumi_coregistrati_alla_t1_bias' dove salva i volumi corretti.

Output filename: <subject>-<mod>-bias.nii.gz

Esempi di naming:
    Input :
        E:\\Datasets\\Volumi_sani_T1_E_FLAIR_1mm_MNI\\sub01\\anat\\skullstripped\\sub-00002_acqsel_FLAIR_to_mni.nii.gz
        E:\\Datasets\\Volumi_sani_T1_E_FLAIR_1mm_MNI\\sub01\\anat\\skullstripped\\sub-00002_acq-iso08_T1w-stripped_to_mni.nii.gz

    Output:
        sub-00002-FLAIR-bias.nii.gz
        sub-00002-T1-bias.nii.gz
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

BASE_DIR = Path(r"E:\Datasets\Volumi_sani_T1_E_FLAIR_1mm_MNI")

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
    """Ritorna tutte le cartelle soggetto: directory che iniziano con 'sub'."""
    if not base.exists():
        raise FileNotFoundError(f"Base non trovata: {base}")
    for d in sorted(base.iterdir()):
        if d.is_dir() and d.name.lower().startswith("sub"):
            yield d


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
        if p.is_dir():
            yield p


def guess_subject_and_modality(path: Path, default_subj: str):
    """
    Estrae:
      - subject: da 'sub-XXXX' nel nome del file (se presente), altrimenti usa default_subj
      - modality: UNA tra 'T1', 'T2', 'FLAIR' cercando il pattern nel nome file (case-insensitive).

    Esempi:
        sub-00002_acqsel_FLAIR_to_mni.nii.gz
            -> subject = sub-00002
            -> modality = FLAIR

        sub-00002_acq-iso08_T1w-stripped_to_mni.nii.gz
            -> subject = sub-00002
            -> modality = T1
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
    #  Nota: qui dentro ora uso la logica dell'esempio SimpleITK N4BiasFieldImageFilter.
    print(f"      [N4/SITK] N4 su: {input_nii}")
    try:
        # Lettura input come float32 (come nell'esempio)
        inputImage = sitk.ReadImage(str(input_nii), sitk.sitkFloat32)
        image = inputImage

        # Maschera: se non fornita, usa Otsu (come nell'esempio)
        maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)

        # shrinkFactor: replica la logica dell'esempio, ma qui fisso a 1 (nessun downsample)
        shrinkFactor = 1
        if shrinkFactor > 1:
            image = sitk.Shrink(
                inputImage, [shrinkFactor] * inputImage.GetDimension()
            )
            maskImage = sitk.Shrink(
                maskImage, [shrinkFactor] * inputImage.GetDimension()
            )

        # Filtro N4
        corrector = sitk.N4BiasFieldCorrectionImageFilter()

        # numberFittingLevels: come nell'esempio, default a 4
        numberFittingLevels = 4

        # Se vuoi imitare la riga:
        # corrector.SetMaximumNumberOfIterations([int(args[5])] * numberFittingLevels)
        # puoi fissare un numero di iterazioni ad es. 50:
        # corrector.SetMaximumNumberOfIterations([50] * numberFittingLevels)

        # Esegue la correzione (su immagine eventualmente shrinkata)
        corrected_image = corrector.Execute(image, maskImage)

        # Calcola il log bias field alla risoluzione piena
        log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)

        # Ricostruisce la versione full-resolution corretta
        corrected_image_full_resolution = inputImage / sitk.Exp(log_bias_field)

        # Salva il risultato finale alla risoluzione originale
        sitk.WriteImage(corrected_image_full_resolution, str(output_nii))

        # Se shrinkFactor > 1 potresti anche salvare l'immagine shrinkata:
        # sitk.WriteImage(corrected_image, "Python-Example-N4BiasFieldCorrection-shrunk.nrrd")

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
    for f in sorted(vol_dir.iterdir()):
        if f.is_file() and is_nifti(f):
            direct_files.append(f)

    if direct_files:
        print(f"      [DEBUG] Trovati {len(direct_files)} file nella cartella principale.")
        return direct_files

    nested = [f for f in vol_dir.rglob("*") if f.is_file() and is_nifti(f)]
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
    print(f"[INFO] OVERWRITE_EXISTING = {OVERWRITE_EXISTING}")
    print("============================================================\n")

    if "ants" not in sys.modules:
        print("❌ ERRORE: modulo 'ants' non disponibile (antspyx).")
        sys.exit(1)

    total_subj = 0
    total_dirs = 0
    total_vols = 0

    # tqdm sui soggetti
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

        # tqdm sulle cartelle skullstripped
        for vol_dir in tqdm(vol_dirs, desc=f"{subj_name} - skullstripped", unit="dir", leave=False):

            # Cartella di output sorella
            out_dir = vol_dir.parent / "volumi_coregistrati_alla_t1_bias"

            # Se devo sovrascrivere, elimino completamente la cartella di output
            if OVERWRITE_EXISTING and out_dir.exists():
                print(f"  [CLEAN] Rimuovo cartella di output esistente: {out_dir}")
                shutil.rmtree(out_dir)

            # (ri)creo la cartella
            out_dir.mkdir(exist_ok=True)

            print(f"\n  [DIR] Input : {vol_dir}")
            print(f"       Output: {out_dir}")

            vols = collect_nifti_files(vol_dir)
            if not vols:
                continue

            # tqdm sui volumi
            for v in tqdm(vols, desc=f"{subj_name} - volumi", unit="vol", leave=False):

                # SKIP se il volume è una maschera
                if "mask" in v.name.lower():
                    print(f"      [SKIP] Volume di maschera (no N4): {v.name}")
                    continue

                subj_id, mod = guess_subject_and_modality(v, default_subj=subj_name)
                out_name = f"{subj_id}-{mod}-bias.nii.gz"
                out_path = out_dir / out_name

                # Se NON sovrascrivo e il file esiste già, salto
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
