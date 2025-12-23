import os
from pathlib import Path
import SimpleITK as sitk
import shutil  # <--- NEW per rimuovere la cartella di output

# proviamo a usare tqdm per la barra di avanzamento
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ===================== CONFIG =====================

BASE_DATASET = Path(r"E:\Datasets\VOLUMI-SANI-1mm")  # root dataset originale
MODE = "affine"
SAVE_TFM = True

HIST_BINS = 50
ITERS = 300
SAMPLING_PCT = 0.25
SHRINK = (4, 2, 1)
SMOOTH = (2, 1, 0)

OUTPUT_FOLDER_NAME = "coregistrati_alla_t1"

# === NUOVO: root parallela dove salvare SOLO i coregistrati ===
PARALLEL_ROOT = BASE_DATASET.parent / f"{BASE_DATASET.name}_coregistrati"

# excel di mappatura soggetti messo nella root parallela
EXCEL_OUT = PARALLEL_ROOT / "coregistrati_mapping.xlsx"

# ===================== FUNZIONI BASE =====================

def read_img(p: Path, pixel_type=sitk.sitkFloat32):
    return sitk.ReadImage(str(p), pixel_type)

def write_img(img: sitk.Image, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(p))
    print(f"[SAVED] {p.resolve()}")

def write_tx(tx: sitk.Transform, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(tx, str(p))
    print(f"[SAVED TX] {p.resolve()}")

def ensure_isotropic_1mm(img: sitk.Image, label: str = "") -> sitk.Image:
    """
    Se lo spacing non è (1,1,1) lo riscampiona con nearest neighbor a 1mm isotropico.
    Stampa sempre lo spacing prima e dopo.
    """
    spacing = img.GetSpacing()
    if label:
        print(f"      📏 Spacing {label} originale: {spacing}")

    # se è già 1mm, stampo e ritorno
    if all(abs(s - 1.0) < 1e-6 for s in spacing):
        print(f"      ✅ {label} già isotropico (1.0, 1.0, 1.0)")
        return img

    orig_size = img.GetSize()
    new_spacing = (1.0, 1.0, 1.0)
    new_size = [
        int(round(orig_size[i] * (spacing[i] / new_spacing[i])))
        for i in range(3)
    ]

    print(f"      ↪️  Resampling {label} a 1mm NN")
    print(f"         - old spacing: {spacing}")
    print(f"         - old size: {orig_size}")
    print(f"         - new size: {new_size}")

    resampled = sitk.Resample(
        img,
        new_size,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        img.GetOrigin(),
        new_spacing,
        img.GetDirection(),
        0,
        img.GetPixelID(),
    )
    print(f"      ✅ Spacing {label} dopo resampling: {resampled.GetSpacing()}")
    return resampled

def build_init_tx(fixed: sitk.Image, moving: sitk.Image, mode: str):
    base = sitk.Euler3DTransform() if mode == "rigid" else sitk.AffineTransform(3)
    return sitk.CenteredTransformInitializer(
        fixed, moving, base, sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

def register(fixed_img: sitk.Image, moving_img: sitk.Image, mode="rigid"):
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=HIST_BINS)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(SAMPLING_PCT)
    R.SetInterpolator(sitk.sitkLinear)

    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=ITERS,
        relaxationFactor=0.5
    )
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel(list(SHRINK))
    R.SetSmoothingSigmasPerLevel(list(SMOOTH))
    try:
        R.SmoothingSigmasAreSpecifiedInPhysicalUnits(False)
    except AttributeError:
        R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOff()

    init_tx = build_init_tx(fixed_img, moving_img, mode)
    R.SetInitialTransform(init_tx, inPlace=False)
    final_tx = R.Execute(fixed_img, moving_img)
    return final_tx, R.GetMetricValue()

def resample_with_tx(moving: sitk.Image, fixed_like: sitk.Image, tx: sitk.Transform,
                     interp=sitk.sitkLinear, default_val=0.0):
    return sitk.Resample(moving, fixed_like, tx, interp, default_val, moving.GetPixelID())

# ===================== FIND MODALITIES =====================
def find_modalities_in_folder(folder: Path):
    """
    Trova T1, T2, FLAIR, PD nella cartella.

    T1:
      1) se esistono file con 'mprage' e 'scic' nel nome (case-insensitive) -> usa quelli
      2) altrimenti file con 'mprage'
      3) altrimenti file con 't1w'

    T2 / FLAIR / PD:
      - se esistono file con 'scic' nel nome (case-insensitive) -> usa quelli
      - altrimenti il primo T2/FLAIR/PD trovato.

    ⚠️ Esclude tutti i file che contengono 'mask' nel nome.
    """

    # ----- T1 -----
    t1_mprage_scic = None   # mprage + scic (priorità massima)
    t1_mprage_any = None    # mprage senza scic
    t1_any = None           # t1w generico

    # ----- T2 / FLAIR / PD -----
    t2_scic = None
    t2_any = None
    flair_scic = None
    flair_any = None
    pd_scic = None
    pd_any = None

    for p in folder.glob("*.nii*"):
        name = p.name.lower()

        # 🔴 SALTA tutti i file che hanno 'mask' nel nome
        if "mask" in name:
            continue

        # ---------- T1 ----------
        if "mprage" in name:
            if "scic" in name:
                if t1_mprage_scic is None:
                    t1_mprage_scic = p
            else:
                if t1_mprage_any is None:
                    t1_mprage_any = p
            continue

        if "t1w" in name:
            if t1_any is None:
                t1_any = p
            continue

        # ---------- T2 ----------
        if "t2w" in name and "t2star" not in name and "t2*" not in name:
            if "scic" in name:
                if t2_scic is None:
                    t2_scic = p
            else:
                if t2_any is None:
                    t2_any = p
            continue

        # ---------- FLAIR ----------
        if "flair" in name:
            if "scic" in name:
                if flair_scic is None:
                    flair_scic = p
            else:
                if flair_any is None:
                    flair_any = p
            continue

        # ---------- PD ----------
        # semplice: 'pd' nel nome, evitando di confonderlo con t1/t2/flair
        if "pd" in name and "t1" not in name and "t2" not in name and "flair" not in name:
            if "scic" in name:
                if pd_scic is None:
                    pd_scic = p
            else:
                if pd_any is None:
                    pd_any = p
            continue

    # scelta finale con le priorità richieste
    t1 = (
        t1_mprage_scic
        if t1_mprage_scic is not None
        else (t1_mprage_any if t1_mprage_any is not None else t1_any)
    )
    t2 = t2_scic if t2_scic is not None else t2_any
    flair = flair_scic if flair_scic is not None else flair_any
    pd = pd_scic if pd_scic is not None else pd_any

    return t1, t2, flair, pd


# ===================== CHECK GIÀ FATTO (sulla cartella parallela) =====================

def already_processed(parallel_anat_dir: Path, t1_name: str | None) -> bool:
    """
    Rimane definita ma non viene più usata per skippare, ora sovrascriviamo sempre
    la cartella di output se esiste.
    """
    out_dir = parallel_anat_dir / OUTPUT_FOLDER_NAME
    if not out_dir.is_dir():
        return False
    if t1_name is None:
        return True
    return (out_dir / t1_name).is_file()

# ===================== EXCEL / CSV =====================

def save_subject_mapping(mapping: list[tuple[int, str]], xlsx_path: Path):
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "mappatura"
        ws.append(["new_sub_id", "original_folder"])
        for idx, original in mapping:
            ws.append([f"sub-{idx:02d}", original])
        wb.save(str(xlsx_path))
        print(f"📑 Mappatura soggetti salvata in: {xlsx_path}")
    except ImportError:
        csv_path = xlsx_path.with_suffix(".csv")
        import csv
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["new_sub_id", "original_folder"])
            for idx, original in mapping:
                w.writerow([f"sub-{idx:02d}", original])
        print(f"📑 openpyxl non disponibile, mappatura salvata in CSV: {csv_path}")

# ===================== PROCESS ANAT =====================
def process_anat_folder(src_anat_dir: Path, dst_anat_dir: Path):
    if not src_anat_dir.is_dir():
        print(f"   ❌ {src_anat_dir} non è una cartella, salto.")
        return

    # PRIORITÀ INPUT:
    # 1) anat/preproc_out    (output di Resampling.py)
    # 2) anat/3dVOL
    # 3) anat direttamente
    preproc_dir = src_anat_dir / "preproc_out"
    vol_dir = src_anat_dir / "3dVOL"

    if preproc_dir.is_dir():
        work_dir = preproc_dir
        print(f"   📁 Uso la cartella preproc_out (input ricampionati): {work_dir}")
    elif vol_dir.is_dir():
        work_dir = vol_dir
        print(f"   📁 Uso la cartella 3dVOL (input): {work_dir}")
    else:
        work_dir = src_anat_dir
        print(f"   📁 Uso la cartella anat direttamente (input): {work_dir}")

    # 🔹 ORA prendiamo anche la PD
    t1_p, t2_p, flair_p, pd_p = find_modalities_in_folder(work_dir)

    if t1_p is None:
        print(f"   ❌ Nessuna T1 (t1w / mprage) trovata in {work_dir}, salto.")
        return

    print(f"   ✅ Trovata T1: {t1_p.name}")
    if t2_p:
        print(f"   ✅ Trovata T2 (preferendo 'scic' se presente): {t2_p.name}")
    else:
        print(f"   ℹ️  Nessuna T2 trovata.")
    if flair_p:
        print(f"   ✅ Trovata FLAIR (preferendo 'scic' se presente): {flair_p.name}")
    else:
        print(f"   ℹ️  Nessuna FLAIR trovata.")
    if pd_p:
        print(f"   ✅ Trovata PD (preferendo 'scic' se presente): {pd_p.name}")
    else:
        print(f"   ℹ️  Nessuna PD trovata.")

    # === sovrascrivi sempre la cartella di output eliminandola se esiste ===
    out_dir = dst_anat_dir / OUTPUT_FOLDER_NAME
    if out_dir.is_dir():
        print(f"   🗑️  Rimuovo cartella di output esistente: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"   📂 Cartella di output parallela: {out_dir}")

    # -------- T1 (fixed) --------
    t1_img = read_img(t1_p)
    t1_img = ensure_isotropic_1mm(t1_img, label="T1")
    write_img(t1_img, out_dir / t1_p.name)

    # -------- T2 → T1 --------
    if t2_p is not None:
        t2_img = read_img(t2_p)
        print("   🔁 Registro T2 → T1 ...")
        t2_tx, t2_metric = register(t1_img, t2_img, mode=MODE)
        print(f"   📏 Metrica T2→T1: {t2_metric:.6f}")
        t2_coreg = resample_with_tx(t2_img, t1_img, t2_tx,
                                    interp=sitk.sitkLinear, default_val=0.0)
        t2_coreg = ensure_isotropic_1mm(t2_coreg, label="T2_coreg")
        write_img(t2_coreg, out_dir / t2_p.name)
        if SAVE_TFM:
            write_tx(t2_tx, out_dir / f"{t2_p.stem}_to_T1.tfm")

    # -------- FLAIR → T1 --------
    if flair_p is not None:
        flair_img = read_img(flair_p)
        print("   🔁 Registro FLAIR → T1 ...")
        flair_tx, flair_metric = register(t1_img, flair_img, mode=MODE)
        print(f"   📏 Metrica FLAIR→T1: {flair_metric:.6f}")
        flair_coreg = resample_with_tx(flair_img, t1_img, flair_tx,
                                       interp=sitk.sitkLinear, default_val=0.0)
        flair_coreg = ensure_isotropic_1mm(flair_coreg, label="FLAIR_coreg")
        write_img(flair_coreg, out_dir / flair_p.name)
        if SAVE_TFM:
            write_tx(flair_tx, out_dir / f"{flair_p.stem}_to_T1.tfm")

    # -------- PD → T1 --------
    if pd_p is not None:
        pd_img = read_img(pd_p)
        print("   🔁 Registro PD → T1 ...")
        pd_tx, pd_metric = register(t1_img, pd_img, mode=MODE)
        print(f"   📏 Metrica PD→T1: {pd_metric:.6f}")
        pd_coreg = resample_with_tx(pd_img, t1_img, pd_tx,
                                    interp=sitk.sitkLinear, default_val=0.0)
        pd_coreg = ensure_isotropic_1mm(pd_coreg, label="PD_coreg")
        write_img(pd_coreg, out_dir / pd_p.name)
        if SAVE_TFM:
            write_tx(pd_tx, out_dir / f"{pd_p.stem}_to_T1.tfm")


# ===================== MAIN =====================

def main():
    base = BASE_DATASET.resolve()
    if not base.exists():
        print(f"❌ Base dataset non trovata: {base}")
        return

    PARALLEL_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"🔎 Scansiono i soggetti in: {base}")
    print(f"📦 Salverò i coregistrati in: {PARALLEL_ROOT}")

    sub_dirs = [p for p in base.iterdir() if p.is_dir() and p.name.lower().startswith("sub")]

    def sub_key(p: Path):
        name = p.name.lower()
        rest = name[3:]
        rest = rest.lstrip("-")
        try:
            return int(rest)
        except ValueError:
            return 999999

    sub_dirs = sorted(sub_dirs, key=sub_key)

    if not sub_dirs:
        print("⚠️ Nessuna cartella che inizi con 'sub'")
        return

    total = len(sub_dirs)
    mapping = []

    # se abbiamo tqdm, la usiamo
    iterator = sub_dirs
    if tqdm is not None:
        iterator = tqdm(sub_dirs, desc="Coregistrazione soggetti", unit="soggetto")

    for idx, sub_dir in enumerate(iterator, start=1):
        if tqdm is None:
            print(f"\n📂 Soggetto ({idx}/{total}): {sub_dir.name}")
        else:
            print(f"\n📂 Soggetto: {sub_dir.name}")

        mapping.append((idx, sub_dir.name))

        parallel_sub_dir = PARALLEL_ROOT / sub_dir.name
        parallel_sub_dir.mkdir(parents=True, exist_ok=True)

        ses_dirs = list(sub_dir.glob("ses-*"))
        if not ses_dirs:
            print("   ⚠️ Nessuna cartella 'ses-*' trovata.")
            src_anat_dir = sub_dir / "anat"
            if src_anat_dir.is_dir():
                print(f"   ✅ Trovata cartella anat: {src_anat_dir}")
                dst_anat_dir = parallel_sub_dir / "anat"
                process_anat_folder(src_anat_dir, dst_anat_dir)
            else:
                print(f"   ❌ Nessuna cartella anat trovata in {sub_dir}")
            continue

        for ses_dir in ses_dirs:
            print(f"   📁 Sessione: {ses_dir.name}")
            src_anat_dir = ses_dir / "anat"
            if src_anat_dir.is_dir():
                print(f"      ✅ Trovata anat: {src_anat_dir}")
                dst_anat_dir = parallel_sub_dir / ses_dir.name / "anat"
                process_anat_folder(src_anat_dir, dst_anat_dir)
            else:
                print(f"      ❌ Nessuna cartella anat trovata in {ses_dir}")

    save_subject_mapping(mapping, EXCEL_OUT)

    print("\n✅ Coregistrazione completata per tutte le cartelle trovate.")

if __name__ == "__main__":
    main()

