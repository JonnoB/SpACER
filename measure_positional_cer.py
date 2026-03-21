"""
Measure CER degradation from inferred character positions.

For each of 750 randomly-generated crop boxes (25 size combos x 30 repeats),
the same box is applied to all 450 pages simultaneously using vectorised
numpy operations, giving 337,500 (page, crop) CER samples.

No edit-distance is needed: the CER reduces to pure set arithmetic since
substitutions are always 0. A character is "captured" if its centre point
falls within the crop box. Characters unique to the GT crop are deletions;
characters unique to the inferred crop are insertions.

Output: data/spatial_uncertainty/crop_cer_results.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/spatial_uncertainty")
OUTPUT_PATH = DATA_DIR / "crop_cer_results.parquet"

SEED = 42
N_REPEATS = 30
FRACTIONS = [0.10, 0.20, 0.30, 0.40, 0.50]

# Actual page extent (pixels) — derived from data; all pages identical
PAGE_W = 963.685
PAGE_H = 1527.931

VARIANTS = ["line", "word", "para"]


# ---------------------------------------------------------------------------
# Phase 1: Load and prepare data
# ---------------------------------------------------------------------------
def load_data():
    print("Loading parquets...")

    # Columns needed from GT
    gt_cols = ["page_id", "char_id", "char_text", "x", "y", "w", "h",
               "font", "alignment", "n_columns"]
    gt = pd.read_parquet(DATA_DIR / "characters_gt.parquet", columns=gt_cols)

    # Drop spaces — consistent with existing analysis
    gt = gt[gt["char_text"] != " "].reset_index(drop=True)

    # Precompute GT centre points
    gt["cx"] = gt["x"] + gt["w"] / 2.0
    gt["cy"] = gt["y"] + gt["h"] / 2.0

    # Encode page_id as integer index 0..N_pages-1
    page_ids = gt["page_id"].unique()
    page_id_to_idx = {pid: i for i, pid in enumerate(page_ids)}
    n_pages = len(page_ids)
    gt["page_idx"] = gt["page_id"].map(page_id_to_idx).astype(np.int32)

    # Per-page metadata (one row per page)
    page_meta = (
        gt[["page_id", "page_idx", "font", "alignment", "n_columns"]]
        .drop_duplicates("page_id")
        .sort_values("page_idx")
        .reset_index(drop=True)
    )

    # Load inferred variants — only need char_id, x, y, w, h
    inf_cols = ["char_id", "x", "y", "w", "h"]
    inferred = {}
    for v in VARIANTS:
        print(f"  Loading characters_{v}...")
        df = pd.read_parquet(DATA_DIR / f"characters_{v}.parquet", columns=inf_cols)
        # Merge onto GT to align rows and filter spaces
        merged = gt[["char_id", "page_idx"]].merge(df, on="char_id", how="left")
        merged["cx"] = merged["x"] + merged["w"] / 2.0
        merged["cy"] = merged["y"] + merged["h"] / 2.0
        inferred[v] = merged

    return gt, inferred, page_meta, page_id_to_idx, n_pages


# ---------------------------------------------------------------------------
# Phase 2: Generate all crop boxes
# ---------------------------------------------------------------------------
def generate_crops(rng: np.random.Generator) -> pd.DataFrame:
    records = []
    crop_id = 0
    for w_frac in FRACTIONS:
        for h_frac in FRACTIONS:
            cw = w_frac * PAGE_W
            ch = h_frac * PAGE_H
            # Random origins so crop stays within page
            x0s = rng.uniform(0, PAGE_W - cw, size=N_REPEATS)
            y0s = rng.uniform(0, PAGE_H - ch, size=N_REPEATS)
            for repeat in range(N_REPEATS):
                records.append({
                    "crop_id": crop_id,
                    "w_frac": w_frac,
                    "h_frac": h_frac,
                    "repeat": repeat,
                    "x0": x0s[repeat],
                    "y0": y0s[repeat],
                    "x1": x0s[repeat] + cw,
                    "y1": y0s[repeat] + ch,
                })
                crop_id += 1
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Phase 3: Vectorised CER computation
# ---------------------------------------------------------------------------
def compute_cer_for_crop(
    x0: float, y0: float, x1: float, y1: float,
    gt_cx: np.ndarray, gt_cy: np.ndarray,
    inf_data: dict,          # variant -> (cx array, cy array)
    page_idx: np.ndarray,
    n_pages: int,
) -> dict:
    """
    Returns dict: variant -> length-n_pages float array of CER values.
    Uses np.bincount for O(n) aggregation — no Python loop over pages.
    """
    # GT mask
    gt_in = (
        (gt_cx >= x0) & (gt_cx <= x1) &
        (gt_cy >= y0) & (gt_cy <= y1)
    ).astype(np.float64)

    n_gt = np.bincount(page_idx, weights=gt_in, minlength=n_pages)

    results = {}
    for v, (inf_cx, inf_cy) in inf_data.items():
        inf_in = (
            (inf_cx >= x0) & (inf_cx <= x1) &
            (inf_cy >= y0) & (inf_cy <= y1)
        ).astype(np.float64)

        n_inf  = np.bincount(page_idx, weights=inf_in,           minlength=n_pages)
        n_both = np.bincount(page_idx, weights=gt_in * inf_in,   minlength=n_pages)

        deletions  = n_gt - n_both
        insertions = n_inf - n_both

        # Avoid divide-by-zero for empty crops (CER = 0)
        cer = np.where(n_gt > 0, (deletions + insertions) / n_gt, 0.0)
        results[v] = cer

    return results, n_gt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)

    gt, inferred, page_meta, _, n_pages = load_data()
    crops = generate_crops(rng)
    print(f"Generated {len(crops)} crop boxes across {n_pages} pages "
          f"= {len(crops) * n_pages:,} samples")

    # Extract numpy arrays once — avoids per-iteration pandas overhead
    gt_cx    = gt["cx"].to_numpy(np.float64)
    gt_cy    = gt["cy"].to_numpy(np.float64)
    page_idx = gt["page_idx"].to_numpy(np.int32)

    inf_arrays = {
        v: (df["cx"].to_numpy(np.float64), df["cy"].to_numpy(np.float64))
        for v, df in inferred.items()
    }

    # Pre-allocate output arrays
    n_crops  = len(crops)
    all_crop_id  = np.empty(n_crops * n_pages, dtype=np.int32)
    all_page_idx = np.tile(np.arange(n_pages, dtype=np.int32), n_crops)
    all_n_gt     = np.empty(n_crops * n_pages, dtype=np.float32)
    all_cer      = {v: np.empty(n_crops * n_pages, dtype=np.float32) for v in VARIANTS}

    print("Computing CER for all crops...")
    crop_rows = crops.to_numpy()  # faster iteration than iterrows
    crop_cols = list(crops.columns)
    ci = {c: crop_cols.index(c) for c in crop_cols}

    for i, row in enumerate(tqdm(crop_rows, unit="crop")):
        start = i * n_pages
        end   = start + n_pages

        cer_dict, n_gt = compute_cer_for_crop(
            x0=row[ci["x0"]], y0=row[ci["y0"]],
            x1=row[ci["x1"]], y1=row[ci["y1"]],
            gt_cx=gt_cx, gt_cy=gt_cy,
            inf_data=inf_arrays,
            page_idx=page_idx,
            n_pages=n_pages,
        )

        all_crop_id[start:end]  = int(row[ci["crop_id"]])
        all_n_gt[start:end]     = n_gt.astype(np.float32)
        for v in VARIANTS:
            all_cer[v][start:end] = cer_dict[v].astype(np.float32)

    # ---------------------------------------------------------------------------
    # Phase 4: Assemble and save results
    # ---------------------------------------------------------------------------
    print("Assembling results...")

    # Expand crop metadata to match (n_crops * n_pages)
    crops_expanded = crops.loc[crops.index.repeat(n_pages)].reset_index(drop=True)

    # Expand page metadata to match (tiled n_crops times)
    page_meta_expanded = pd.concat(
        [page_meta] * n_crops, ignore_index=True
    )

    results = pd.DataFrame({
        "crop_id":     all_crop_id,
        "page_id":     page_meta_expanded["page_id"].values,
        "font":        page_meta_expanded["font"].values,
        "alignment":   page_meta_expanded["alignment"].values,
        "n_columns":   page_meta_expanded["n_columns"].values,
        "w_frac":      crops_expanded["w_frac"].values.astype(np.float32),
        "h_frac":      crops_expanded["h_frac"].values.astype(np.float32),
        "repeat":      crops_expanded["repeat"].values.astype(np.int16),
        "x0":          crops_expanded["x0"].values.astype(np.float32),
        "y0":          crops_expanded["y0"].values.astype(np.float32),
        "n_gt_chars":  all_n_gt,
        "cer_line":    all_cer["line"],
        "cer_word":    all_cer["word"],
        "cer_para":    all_cer["para"],
    })

    # Drop crops where GT had zero characters (no meaningful CER)
    results = results[results["n_gt_chars"] > 0].reset_index(drop=True)

    print(f"Saving {len(results):,} rows to {OUTPUT_PATH} ...")
    results.to_parquet(OUTPUT_PATH, index=False)
    print("Done.")
    print(results.describe())


if __name__ == "__main__":
    main()
