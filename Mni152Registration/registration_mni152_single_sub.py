import os
from pathlib import Path
import glob
import ants
import sys

# =========================
# CONFIG
# =========================

# Cartella di input del soggetto (quella che mi hai dato)
SKULL_FOLDER = Path(
    r"C:\Users\Stefano\Desktop\Stefano\ImmaginiTesi\sub-ON04111(check t2)\volumes_generated_no_bandoni"
)

# Template MNI T1
MNI_T1 = (
    r"C:\Users\Stefano\Desktop\Stefano\Codice_skull\Auto-SkullStripping\Mni152Registration\mni152\icbm_avg_152_t1_tal_nlin_symmetric_VI_mask.nii"
)

# Tipo di registrazione mod → T1
TYPE_M2T1 = "Rigid"  # "Rigid" | "AffineFast" | "Affine"

# Dove salvare le trasformazioni (le metto a livello soggetto)
SUBJECT_DIR = SKULL_FOLDER.parents[2]  # ...\sub08(check flair e t1)
TFM_ROOT = SUBJECT_DIR / "_Transforms_to_MNI"
TFM_ROOT.mkdir(parents=True, exist_ok=True)


# =========================
# Helpers nomi / file
# =========================

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
    """
    Cerca un file di una certa modalità in una cartella.

    Modalità supportate:
      - T1  (t1, t1w, mprage, mp2rage)
      - FLAIR
      - T2
      - PD
    """
    files = list(folder.glob("*.nii")) + list(folder.glob("*.nii.gz"))
    modality = modality.upper()

    for f in files:
        name = f.name.lower()

        # escludi maschere
        if "mask" in name:
            continue

        # caso speciale: file che contengono sia flair che t2 -> considerali flair
        if "flair" in name and "t2" in name:
            if modality == "FLAIR":
                return f
            else:
                continue

        # ===============================
        # T1
        # ===============================
        if modality == "T1":
            if (
                "t1w" in name
                or "mprage" in name
                or "mp2rage" in name
                or name.startswith("t1")
                or "_t1" in name
            ):
                return f

        # ===============================
        # FLAIR
        # ===============================
        elif modality == "FLAIR":
            if "flair" in name:
                return f

        # ===============================
        # T2
        # ===============================
        elif modality == "T2":
            if (
                "t2" in name
                and "t2star" not in name
                and "t2*" not in name
                and "flair" not in name
            ):
                return f

        # ===============================
        # PD
        # ===============================
        elif modality == "PD":
            if "pd" in name or "proton" in name:
                return f

    return None


# =========================
# Trasformazioni
# =========================

def t1_outprefix_for_transforms(t1_name: str) -> Path:
    base = t1_name
    if base.endswith(".nii.gz"):
        base = base[:-7]
    elif base.endswith(".nii"):
        base = base[:-4]
    return TFM_ROOT / f"{base}_T1toMNI_"


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
    warp = next(
        (p for p in fwd if p.endswith("Warp.nii.gz") or p.endswith("Warp.nii")), None
    )
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


# =========================
# MAIN single-folder
# =========================

def main():
    if not SKULL_FOLDER.exists():
        print(f"❌ La cartella di input non esiste:\n   {SKULL_FOLDER}")
        return 1

    print(f"[INFO] Cartella soggetto (skullstripped):\n       {SKULL_FOLDER}")

    # useremo la stessa cartella come output
    dst_skull = SKULL_FOLDER
    print(f"[INFO] Output (stessa cartella, con suffisso _to_mni):\n       {dst_skull}")

    # carico template MNI una volta
    mni_img = ants.image_read(MNI_T1)

    # ========== 1) T1 ==========
    t1_path = find_modality_file(SKULL_FOLDER, "T1")
    if t1_path is None:
        print("⚠️ Nessuna T1/MPRAGE trovata in questa cartella, esco.")
        return 1

    print(f"✅ T1 trovata: {t1_path.name}")

    tfm_prefix = t1_outprefix_for_transforms(t1_path.name)
    t1_dst = dst_skull / with_to_mni_suffix(t1_path.name)

    t1_img = ants.image_read(str(t1_path))
    t1_to_mni_chain = None

    if t1_dst.exists():
        print(f"⏩ T1 già registrata: {t1_dst.name}")
        saved = find_saved_t1_to_mni_transforms(tfm_prefix)
        if saved:
            print(f"🔄 Ri-uso trasformazioni già salvate: {saved}")
            t1_to_mni_chain = saved
        else:
            print("⚠️ Trasformazioni non trovate, le rigenero...")
            _, _, _, t1_to_mni_chain = register_t1_to_mni_full(
                MNI_T1, str(t1_path), outprefix=tfm_prefix
            )
    else:
        print("▶ Registro T1 → MNI (SyN)...")
        _, _, t1_in_mni, t1_to_mni_chain = register_t1_to_mni_full(
            MNI_T1, str(t1_path), outprefix=tfm_prefix
        )
        ants.image_write(t1_in_mni, str(t1_dst))
        print(f"✅ Salvata T1 in MNI: {t1_dst.name}")

    if not t1_to_mni_chain:
        print("❌ Non ho una catena T1→MNI valida, esco.")
        return 1

    # ========== 2) FLAIR e T2 ==========
    for mod in ("FLAIR", "T2", "PD"):
        mod_path = find_modality_file(SKULL_FOLDER, mod)
        if mod_path is None:
            print(f"ℹ️ {mod} non trovata, salto.")
            continue

        mod_dst = dst_skull / with_to_mni_suffix(mod_path.name)
        if mod_dst.exists():
            print(f"⏩ {mod} già registrata: {mod_dst.name}")
            continue

        print(f"▶ Registro {mod}: {mod_path.name}")

        moving_img, mod2t1_aff = register_mod_to_t1_affine(
            str(mod_path), t1_img, TYPE_M2T1
        )

        full_chain = [*t1_to_mni_chain, mod2t1_aff]
        mod_in_mni = push_to_mni(moving_img, mni_img, full_chain, is_label=False)
        ants.image_write(mod_in_mni, str(mod_dst))
        print(f"✅ Salvata {mod} in MNI: {mod_dst.name}")

    print("\n🎉 Fatto! Registrazione su MNI completata per questa cartella.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
