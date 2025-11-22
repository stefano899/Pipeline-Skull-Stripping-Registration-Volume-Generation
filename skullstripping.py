#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import argparse
import SimpleITK as sitk
import shutil
import sys

# proviamo a usare tqdm per la progress bar
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ---------------------------------------------------------
# helper
# ---------------------------------------------------------

def is_nifti(p: Path) -> bool:
    return p.is_file() and (p.name.lower().endswith(".nii") or p.name.lower().endswith(".nii.gz"))

def stem_wo_nii(name: str) -> str:
    low = name.lower()
    if low.endswith(".nii.gz"):
        return name[:-7]
    if low.endswith(".nii"):
        return name[:-4]
    return Path(name).stem

# ---------------------------------------------------------
# apply mask (adatta la maschera all’immagine se serve)
# ---------------------------------------------------------

def apply_mask_to_image(img_path: Path, mask_path: Path, out_path: Path, outside_value=0):
    img  = sitk.ReadImage(str(img_path))
    mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)

    # se le griglie non coincidono, porta la mask sull'immagine
    if (
        img.GetSize()      != mask.GetSize() or
        img.GetSpacing()   != mask.GetSpacing() or
        img.GetOrigin()    != mask.GetOrigin() or
        img.GetDirection() != mask.GetDirection()
    ):
        mask = sitk.Resample(
            mask,
            img,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8
        )

    binmask = sitk.Cast(mask > 0, sitk.sitkUInt8)
    masked = sitk.Mask(img, binmask, outside_value)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(masked, str(out_path))
    print(f"[SAVED] {out_path.resolve()}")
    return out_path

# ---------------------------------------------------------
# ROBEX
# ---------------------------------------------------------

def _quote_if_needed(s: str) -> str:
    return f"\"{s}\"" if (" " in s or "(" in s or ")" in s) else s

def run_robex_single(
    robex_dir: Path,
    t1_path: Path,
    out_stripped: Path,
    out_mask: Path,
    seed: Optional[int]=None,
    log_file: Optional[Path]=None
) -> int:
    robex_dir = robex_dir.resolve()
    bat = (robex_dir / "runROBEX.bat").resolve()
    if not bat.exists():
        raise FileNotFoundError(f"Batch non trovato: {bat}")

    out_stripped.parent.mkdir(parents=True, exist_ok=True)
    out_mask.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["cmd.exe", "/C", str(bat), str(t1_path.resolve()), str(out_stripped.resolve()), str(out_mask.resolve())]
    if seed is not None:
        cmd.append(str(seed))

    print(f"[DEBUG] CWD={robex_dir}")
    print("[DEBUG] CMD=" + " ".join(_quote_if_needed(c) for c in cmd))

    if log_file is not None:
        with log_file.open("a", encoding="utf-8") as L:
            print(f"[DEBUG] CWD={robex_dir}", file=L)
            print("[DEBUG] CMD=" + " ".join(_quote_if_needed(c) for c in cmd), file=L)

    res = subprocess.run(cmd, shell=False, cwd=str(robex_dir), capture_output=True, text=True)

    if res.stdout:
        print(res.stdout, end="")
        if log_file is not None:
            with log_file.open("a", encoding="utf-8") as L:
                print(res.stdout, end="", file=L)
    if res.stderr:
        print(res.stderr, end="")
        if log_file is not None:
            with log_file.open("a", encoding="utf-8") as L:
                print(res.stderr, end="", file=L)

    return res.returncode

# ---------------------------------------------------------
# trova T1 / T2 / FLAIR dentro una cartella
# ---------------------------------------------------------

def find_subject_files_in(folder: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in folder.rglob("*.nii*"):
        name = p.name.lower()
        if ("t1w" in name or "mprage" in name) and "T1" not in out:
            out["T1"] = p
        elif "t2w" in name and "t2star" not in name and "t2*" not in name and "T2" not in out:
            out["T2"] = p
        elif "flair" in name and "FLAIR" not in out:
            out["FLAIR"] = p
        if len(out) == 3:
            break
    return out

# ---------------------------------------------------------
# trova tutte le cartelle anat di un soggetto
# ---------------------------------------------------------

def get_all_anat_dirs(subj_dir: Path) -> List[Path]:
    anat_dirs: List[Path] = []

    # 1) anat direttamente
    if (subj_dir / "anat").is_dir():
        anat_dirs.append(subj_dir / "anat")

    # 2) tutte le ses-* con anat
    for ses in subj_dir.glob("ses-*"):
        anat = ses / "anat"
        if anat.is_dir():
            anat_dirs.append(anat)

    return anat_dirs

# ---------------------------------------------------------
# controlla se questa anat è già stata processata
# ---------------------------------------------------------

def already_skullstripped(anat_dir: Path) -> bool:
    out_dir = anat_dir / "skullstripped"
    if not out_dir.is_dir():
        return False
    masks = list(out_dir.glob("*-mask.nii*"))
    return len(masks) > 0

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Skull stripping con ROBEX sui volumi coregistrati (se presenti), per TUTTE le anat di ogni soggetto."
    )
    ap.add_argument("--robex_dir", required=True, help="Cartella che contiene runROBEX.bat")
    ap.add_argument("--subjects_root", required=True, help="Radice con i soggetti (sub01, sub185, ...)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--outside_value", type=float, default=0.0)
    args = ap.parse_args()

    robex_dir = Path(args.robex_dir).resolve()
    subj_root = Path(args.subjects_root).resolve()

    log = subj_root / f"robex_applymask_{datetime.now():%Y%m%d_%H%M%S}.log"
    with log.open("w", encoding="utf-8") as L:
        L.write(f"[INFO] start {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    subjects = [d for d in subj_root.iterdir() if d.is_dir()]
    print(f"[INFO] Soggetti trovati: {len(subjects)}")

    ok = fail = 0

    # ordina tipo sub1, sub2, ..., sub100
    def sub_key(p: Path):
        name = p.name.lower()
        if name.startswith("sub"):
            rest = name[3:].lstrip("-")
            try:
                return int(rest)
            except ValueError:
                return 999999
        return 999999

    subjects = sorted(subjects, key=sub_key)
    total_subj = len(subjects)

    # se abbiamo tqdm, la usiamo
    subj_iter = subjects
    if tqdm is not None:
        subj_iter = tqdm(subjects, desc="Skullstripping", unit="soggetto")

    for idx, subj in enumerate(subj_iter, start=1):
        if tqdm is None:
            print(f"\n[{idx}/{total_subj}] === {subj.name} ===")
        else:
            print(f"\n=== {subj.name} ===")

        anat_dirs = get_all_anat_dirs(subj)
        if not anat_dirs:
            print("   ❌ Nessuna cartella anat trovata in questo soggetto, salto.")
            continue

        for anat_dir in anat_dirs:
            print(f"   👉 ANAT: {anat_dir}")

            # se l'hai già fatto qui, salta
            if already_skullstripped(anat_dir):
                print("      ✅ Skullstripping già presente, salto questa anat.")
                continue

            # se c'è la cartella coreg, lavoriamo lì
            coreg_dir = anat_dir / "coregistrati_alla_t1"
            if coreg_dir.is_dir():
                print(f"      📁 Uso i volumi coregistrati: {coreg_dir}")
                files = find_subject_files_in(coreg_dir)
                src_folder = coreg_dir
            else:
                print(f"      ⚠️ Nessuna 'coregistrati_alla_t1', uso {anat_dir}")
                files = find_subject_files_in(anat_dir)
                src_folder = anat_dir

            if "T1" not in files:
                msg = f"      ❌ T1/MPRAGE non trovato in {src_folder}"
                print(msg)
                with log.open("a", encoding="utf-8") as L:
                    print(msg, file=L)
                continue

            out_dir = anat_dir / "skullstripped"
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"      📂 Output: {out_dir}")

            t1_path = files["T1"]
            print(f"      ✅ T1: {t1_path.name}")

            tmp_strip = out_dir / "_tmp_stripped.nii.gz"
            tmp_mask  = out_dir / "_tmp_mask.nii.gz"

            rc = run_robex_single(
                robex_dir=robex_dir,
                t1_path=t1_path,
                out_stripped=tmp_strip,
                out_mask=tmp_mask,
                seed=args.seed,
                log_file=log,
            )

            # ROBEX può salvare .nii
            if not tmp_strip.exists():
                alt_strip = tmp_strip.with_suffix("")
            else:
                alt_strip = None
            if not tmp_mask.exists():
                alt_mask = tmp_mask.with_suffix("")
            else:
                alt_mask = None

            if (rc != 0) or (not tmp_mask.exists() and not (alt_mask and alt_mask.exists())):
                fail += 1
                print("      ❌ ROBEX non ha prodotto i file attesi.")
                continue

            real_mask = tmp_mask if tmp_mask.exists() else alt_mask

            base_t1 = stem_wo_nii(t1_path.name)
            final_mask = out_dir / f"{base_t1}-mask.nii.gz"
            shutil.move(str(real_mask), str(final_mask))

            final_t1_stripped = out_dir / f"{base_t1}-stripped.nii.gz"
            apply_mask_to_image(t1_path, final_mask, final_t1_stripped, outside_value=args.outside_value)

            if tmp_strip.exists():
                tmp_strip.unlink(missing_ok=True)

            # altre modalità
            if "T2" in files:
                t2_path = files["T2"]
                out_t2 = out_dir / t2_path.name
                apply_mask_to_image(t2_path, final_mask, out_t2, outside_value=args.outside_value)

            if "FLAIR" in files:
                flair_path = files["FLAIR"]
                out_flair = out_dir / flair_path.name
                apply_mask_to_image(flair_path, final_mask, out_flair, outside_value=args.outside_value)

            ok += 1
            with log.open("a", encoding="utf-8") as L:
                print(f"[OK] {subj.name} | anat={anat_dir}", file=L)

    print(f"\n[DONE] OK={ok} FAIL={fail}")
    print(f"[LOG] {log}")
    print("✅ Finito.")

if __name__ == "__main__":
    main()

# python skullstrip_robex.py --robex_dir C:\ROBEX --subjects_root E:\Datasets\...