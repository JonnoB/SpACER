import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # DocBank — select a working subset

    Randomly selects 300 pages from the full DocBank annotation set
    (`data/docbank/DocBank_500K_txt/*.txt`, extracted from the official
    `DocBank_500K_txt.zip`) and writes a file list for `7z` so only the
    matching page images need to be pulled out of the split
    `DocBank_500K_ori_img.zip.001..010` archive.

    This notebook only selects pages and writes the file list — it does not
    run the 7z extraction itself.
    """)
    return


@app.cell
def _():
    import json
    import random
    import shutil
    from pathlib import Path

    import marimo as mo

    return Path, json, mo, random, shutil


@app.cell
def _(Path):
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    TXT_DIR = _REPO_ROOT / "data/docbank/DocBank_500K_txt"
    OUTPUT_DIR = _REPO_ROOT / "data/docbank"
    GT_DIR = OUTPUT_DIR / "gt_word_annotations"
    MSCOCO_ALL_PATH = OUTPUT_DIR / "MSCOCO_Format_Annotation" / "500K_all.json"
    MSCOCO_SUBSET_PATH = OUTPUT_DIR / "mscoco_annotations_subset.json"
    N_SAMPLES = 300
    SEED = 42
    return (
        GT_DIR,
        MSCOCO_ALL_PATH,
        MSCOCO_SUBSET_PATH,
        N_SAMPLES,
        OUTPUT_DIR,
        SEED,
        TXT_DIR,
    )


@app.cell
def _(TXT_DIR):
    # Ground truth (gt) files define which basenames actually exist — use
    # them as the population to sample from rather than assuming coverage.
    basenames = sorted(p.stem for p in TXT_DIR.glob("*.txt"))
    print(f"Found {len(basenames):,} gt files in {TXT_DIR}")
    return (basenames,)


@app.cell
def _(N_SAMPLES, SEED, basenames, random):
    selected_basenames = random.Random(SEED).sample(basenames, N_SAMPLES)
    print(f"Selected {len(selected_basenames)} basenames (seed={SEED})")
    return (selected_basenames,)


@app.cell
def _(OUTPUT_DIR, selected_basenames):
    # Archive stores entries under a folder prefix, e.g.
    # "DocBank_500K_ori_img/{basename}_ori.jpg" (confirmed via `7z l`), so the
    # file list passed to 7z must include it or nothing will match.
    image_filenames = [
        f"DocBank_500K_ori_img/{b}_ori.jpg" for b in selected_basenames
    ]

    filelist_path = OUTPUT_DIR / "image_filelist.txt"
    filelist_path.write_text("\n".join(image_filenames) + "\n")

    basenames_path = OUTPUT_DIR / "selected_basenames.txt"
    basenames_path.write_text("\n".join(selected_basenames) + "\n")

    print(f"Wrote {len(image_filenames)} filenames to {filelist_path}")
    print(f"Wrote {len(selected_basenames)} basenames to {basenames_path}")
    print(
        "Next step (not run here): "
        "cd data/docbank && 7z x DocBank_500K_ori_img.zip.001 @image_filelist.txt -o<output_dir>"
    )
    return basenames_path, filelist_path, image_filenames


@app.cell
def _(GT_DIR, TXT_DIR, selected_basenames, shutil):
    # Copy just the selected pages' raw per-word gt .txt files out of the
    # 500k-file DocBank_500K_txt directory so downstream work doesn't need to
    # touch it. Raw format: word\tx0\ty0\tx1\ty1\tR\tG\tB\tfont\tlabel.
    # scripts/extract_docbank_ssu_text.py turns these into SSU-level regions
    # and plain per-page text under data/docbank/gt_page_texts/.
    GT_DIR.mkdir(parents=True, exist_ok=True)
    for _b in selected_basenames:
        shutil.copy2(TXT_DIR / f"{_b}.txt", GT_DIR / f"{_b}.txt")
    print(f"Copied {len(selected_basenames)} raw word-annotation files to {GT_DIR}")
    return


@app.cell
def _(MSCOCO_ALL_PATH, MSCOCO_SUBSET_PATH, json, selected_basenames):
    # 500K_all.json covers every page (train+valid+test combined), matching
    # the population DocBank_500K_txt was sampled from. It's ~900MB, so this
    # loads it once, filters down to the 300 selected pages, and writes a
    # small subset file — no need to keep the full file open afterwards.
    with open(MSCOCO_ALL_PATH) as _f:
        _coco = json.load(_f)

    _wanted_file_names = {f"{b}_ori.jpg" for b in selected_basenames}

    subset_images = [
        img for img in _coco["images"] if img["file_name"] in _wanted_file_names
    ]
    _wanted_image_ids = {img["id"] for img in subset_images}

    subset_annotations = [
        ann for ann in _coco["annotations"] if ann["image_id"] in _wanted_image_ids
    ]

    coco_subset = {
        "info": _coco["info"],
        "licenses": _coco["licenses"],
        "categories": _coco["categories"],
        "images": subset_images,
        "annotations": subset_annotations,
    }

    with open(MSCOCO_SUBSET_PATH, "w") as _f:
        json.dump(coco_subset, _f)

    print(
        f"Matched {len(subset_images)}/{len(selected_basenames)} selected pages "
        f"({len(subset_annotations)} annotations) -> {MSCOCO_SUBSET_PATH}"
    )
    return coco_subset, subset_annotations, subset_images


if __name__ == "__main__":
    app.run()
