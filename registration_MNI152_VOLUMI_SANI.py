import os
from pathlib import Path
import glob
import ants
import sys

# proviamo a importare tqdm per la progress bar
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ======================================================
# CONFIG
# ======================================================
SRC_ROOT = r"E:\Datasets\VOLUMI-SANI-1mm"          # dataset sorgente
DST_ROOT = r"E:\Datasets\VOLUMI-SANI-1mm_coregistrati"      # dataset di output (stessa struttura)

# metti qui il tuo template MNI
MNI_T1 = r"C:\Users\Stefano\Desktop\Stefano\Codice_skull\Auto-SkullStripping\Mni152Registration\mni152\icbm_avg_152_t1_tal_nlin_symmetric_VI_mask.nii"
# tipo di registrazione mod → T1
TYPE_M2T1 = "Rigid"   # "Rigid" | "AffineFast" | "Affine"

# dove salvo le trasformazioni
TFM_ROOT = Path(DST_ROOT) / "_Transforms"
TFM_ROOT.mkdir(parents=True, exist_ok=True)


# ======================================================
# 1. Trova le cartelle skullstripped
# ======================================================
def discover_skullstripped_folders(root: str):
    """
    Ritorna lista di (subject, rel_path) dove rel_path è il percorso
    dal soggetto fino a 'skullstripped', per es.:

      ("sub01", "anat/skullstripped")
      ("sub100", "ses-01/anat/skullstripped")
    """
    out = []
    root_p = Path(root)
    for subj in root_p.iterdir():
        if not subj.is_dir():
            continue
        if not subj.name.lower().startswith("sub"):
            continue

        # subXX/anat/skullstripped
        anat = subj / "anat"
        if anat.is_dir():
            skull = anat / "volumi_coregistrati_alla_t1"
            if skull.is_dir():
                out.append((subj.name, "anat/skullstripped"))

        # subXX/ses-*/anat/skullstripped
        for ses in subj.glob("ses-*"):
            ses_anat = ses / "anat"
            if ses_anat.is_dir():
                skull = ses_anat / "volumi_coregistrati_alla_t1"
                if skull.is_dir():
                    rel = skull.relative_to(subj)
                    out.append((subj.name, str(rel)))   # es: "ses-01/anat/skullstripped"

    # ora li ordino in modo crescente per numero dopo "sub"
    def sort_key(item):
        subj_name, relp = item
        num_part = subj_name[3:].lstrip("-")
        try:
            return int(num_part)
        except ValueError:
            return 999999
    out = sorted(out, key=sort_key)
    return out


# ======================================================
# 2. Helpers nomi e ricerca file
# ======================================================
def ensure_parent_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def with_to_mni_suffix(name: str) -> str:
    low = name.lower()
    if low.endswith(".nii.gz"):
        base = name[:-7]
        return base + "_to_mni.nii.gz"
    elif low.endswith(".nii"):
        base = name[:-4]
        return base + "_to_mni.nii"
    else:
        return name + "_to_mni"


def find_modality_file(folder: Path, modality: str) -> Path | None:
    files = list(folder.glob("*.nii")) + list(folder.glob("*.nii.gz"))
    modality = modality.upper()

    for f in files:
        name = f.name.lower()

        # escludi maschere
        if "mask" in name:
            continue

        # flair + t2 => flair
        if "flair" in name and "t2" in name:
            if modality == "FLAIR":
                return f
            else:
                continue

        if modality == "T1":
            if "t1w" in name or "mprage" in name:
                return f

        elif modality == "FLAIR":
            if "flair" in name:
                return f

        elif modality == "T2":
            if "t2" in name and "t2star" not in name and "t2*" not in name and "flair" not in name:
                return f

    return None


# ======================================================
# 3. trasformazioni
# ======================================================
def t1_outprefix_for_transforms(subject: str, rel_skull: str, t1_name: str) -> Path:
    rel_flat = rel_skull.replace("\\", "_").replace("/", "_")
    base = t1_name
    if base.endswith(".nii.gz"):
        base = base[:-7]
    elif base.endswith(".nii"):
        base = base[:-4]
    return TFM_ROOT / f"{subject}_{rel_flat}_{base}_T1toMNI_"


def find_saved_t1_to_mni_transforms(prefix: Path) -> list[str] | None:
    warp_candidates = glob.glob(str(prefix) + "*Warp.nii*")
    aff_candidates = glob.glob(str(prefix) + "*GenericAffine.mat")
    warp = sorted(warp_candidates, key=len)[:1]
    aff = sorted(aff_candidates, key=len)[:1]
    if warp and aff:
        return [warp[0], aff[0]]
    return None


def register_t1_to_mni_full(mni_path: str, t1_path: str, outprefix: Path | None = None):
    mni_img = ants.image_read(mni_path)
    t1_img = ants.image_read(t1_path)
    reg_kwargs = dict(fixed=mni_img, moving=t1_img, type_of_transform="SyN")
    if outprefix is not None:
        ensure_parent_dir(outprefix.with_suffix(".sentinel"))
        reg_kwargs["outprefix"] = str(outprefix)
    reg = ants.registration(**reg_kwargs)
    fwd = reg["fwdtransforms"]
    warp = next((p for p in fwd if p.endswith("Warp.nii.gz") or p.endswith("Warp.nii")), None)
    aff = next((p for p in fwd if p.endswith("GenericAffine.mat")), None)
    if aff is None or warp is None:
        raise RuntimeError("SyN non ha prodotto sia warp che affine.")
    return mni_img, t1_img, reg["warpedmovout"], [warp, aff]


def register_mod_to_t1_affine(mod_path: str, t1_img, mode: str):
    moving = ants.image_read(mod_path)
    reg = ants.registration(fixed=t1_img, moving=moving, type_of_transform=mode)
    mat = reg["fwdtransforms"][-1]
    return moving, mat


def push_to_mni(moving_img, mni_img, tf_chain, is_label=False):
    return ants.apply_transforms(
        fixed=mni_img,
        moving=moving_img,
        transformlist=tf_chain,
        interpolator=("nearestNeighbor" if is_label else "linear"),
    )


# ======================================================
# 4. MAIN
# ======================================================
def main():
    src_root = Path(SRC_ROOT)
    dst_root = Path(DST_ROOT)
    dst_root.mkdir(parents=True, exist_ok=True)

    # carico una volta il template MNI
    mni_img = ants.image_read(MNI_T1)

    skull_folders = discover_skullstripped_folders(SRC_ROOT)
    total = len(skull_folders)
    print(f"[INFO] Cartelle skullstripped trovate (ordinate): {total}")

    # se abbiamo tqdm, la usiamo
    iterator = skull_folders
    if tqdm is not None:
        iterator = tqdm(skull_folders, desc="Registrazione su MNI", unit="cartella")

    for i, (subject, rel_skull) in enumerate(iterator, start=1):
        if tqdm is None:
            # fallback: stampa progress semplice
            print(f"\n[{i}/{total}] === {subject} | {rel_skull} ===")
        else:
            # con tqdm stampiamo meno roba, ma teniamo il blocco funzionale
            print(f"\n=== {subject} | {rel_skull} ===")

        # sorgente e destinazione con la STESSA struttura
        src_skull = src_root / subject / rel_skull
        dst_skull = dst_root / subject / rel_skull
        dst_skull.mkdir(parents=True, exist_ok=True)
        print(f"  📁 Output: {dst_skull}")

        # 1) prendo T1
        t1_path = find_modality_file(src_skull, "T1")
        if t1_path is None:
            print("  ⚠️ Nessuna T1/MPRAGE trovata qui, salto questa cartella.")
            continue
        print(f"  ✅ T1 trovata: {t1_path.name}")

        # prefisso trasformazioni
        tfm_prefix = t1_outprefix_for_transforms(subject, rel_skull, t1_path.name)

        # nome di output: stesso nome + _to_mni
        t1_dst = dst_skull / with_to_mni_suffix(t1_path.name)

        # 2) registra T1 → MNI (o riusa)
        t1_to_mni_chain = None
        t1_img = ants.image_read(str(t1_path))

        if t1_dst.exists():
            print(f"  ⏩ T1 già registrata: {t1_dst.name}")
            saved = find_saved_t1_to_mni_transforms(tfm_prefix)
            if saved:
                print(f"  🔄 Ri-uso trasformazioni già salvate: {saved}")
                t1_to_mni_chain = saved
            else:
                print("  ⚠️ Trasformazioni non trovate, le rigenero (solo trasformi)...")
                _, _, _, t1_to_mni_chain = register_t1_to_mni_full(MNI_T1, str(t1_path), outprefix=tfm_prefix)
        else:
            print("  ▶ Registro T1 → MNI (SyN)...")
            _, _, t1_in_mni, t1_to_mni_chain = register_t1_to_mni_full(MNI_T1, str(t1_path), outprefix=tfm_prefix)
            ants.image_write(t1_in_mni, str(t1_dst))
            print(f"  ✅ Salvata T1 in MNI: {t1_dst.name}")

        if not t1_to_mni_chain:
            print("  ❌ Non ho una catena T1→MNI valida, salto le altre modalità.")
            continue

        # 3) FLAIR e T2
        for mod in ("FLAIR", "T2"):
            mod_path = find_modality_file(src_skull, mod)
            if mod_path is None:
                print(f"  ℹ️ {mod} non trovata, salto.")
                continue

            mod_dst = dst_skull / with_to_mni_suffix(mod_path.name)
            if mod_dst.exists():
                print(f"  ⏩ {mod} già registrata: {mod_dst.name}")
                continue

            print(f"  ▶ Registro {mod}: {mod_path.name}")

            # mod → T1
            moving_img, mod2t1_aff = register_mod_to_t1_affine(str(mod_path), t1_img, TYPE_M2T1)

            # catena T1→MNI + mod→T1
            full_chain = [*t1_to_mni_chain, mod2t1_aff]

            mod_in_mni = push_to_mni(moving_img, mni_img, full_chain, is_label=False)
            ants.image_write(mod_in_mni, str(mod_dst))
            print(f"    ✅ Salvata {mod} in MNI: {mod_dst.name}")

    print("\n[TUTTO FATTO] Registrazione su MNI completata con nomi originali + _to_mni e stessa struttura.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

