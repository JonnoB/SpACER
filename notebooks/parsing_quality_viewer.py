import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Page-level parsing-quality viewer

    Pick a dataset, browse pages/models sorted by worst SpACER first (with a
    COTe floor, to surface the "parsing looks geometrically fine but OCR
    still failed" cases), and inspect the original page next to its COTe
    overlay (green = coverage, amber = overlap, red = trespass, purple =
    overlap+trespass, blue = excess).

    The browse table joins the pre-computed `cote_score_cache.parquet`
    (bbox fast-path COTe) with `page_level_cer_comparison.parquet` (CER /
    SpACER / CDD per OCR model) for each dataset; the selected page's
    overlay is rasterized and computed fresh here, since only scalar scores
    exist on disk, not pixel masks.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image

    from cotescore import cote_score
    from cotescore.visualisation import compute_cote_masks, visualize_cote_states
    from cotescore.adapters import boxes_to_gt_ssu_map, boxes_to_pred_masks

    return (
        Image,
        Path,
        boxes_to_gt_ssu_map,
        boxes_to_pred_masks,
        compute_cote_masks,
        cote_score,
        np,
        pd,
        plt,
        visualize_cote_states,
    )


@app.cell
def _(Path):
    REPO_ROOT = Path(__file__).resolve().parent.parent

    DATASETS = {
        "spiritualist": (
            "data/spiritualist/gt_ssu_bboxes.csv",
            "data/results_spiritualist/bboxes",
            "data/spiritualist/spiritualist_images_120dpi",
            "data/spiritualist/cote_score_cache.parquet",
            "data/results_spiritualist/page_level_cer_comparison.parquet",
        ),
        "hiertext": (
            "data/hiertext/gt_ssu_bboxes.csv",
            "data/hiertext_predictions",
            "data/hiertext/hiertext_validation",
            "data/hiertext/cote_score_cache.parquet",
            "data/hiertext/page_level_cer_comparison.parquet",
        ),
        "docbank": (
            "data/docbank/gt_ssu_bboxes.csv",
            "data/docbank/bbox_preds",
            "data/docbank/images_subset",
            "data/docbank/cote_score_cache.parquet",
            "data/docbank/page_level_cer_comparison.parquet",
        ),
    }
    return DATASETS, REPO_ROOT


@app.cell
def _(DATASETS, Path, REPO_ROOT, pd):
    def load_dataset(name):
        """Return (gt_df, pred_df, page_to_filename) for a dataset.

        pred_df has one row per predicted box, tagged with a `parsing_model`
        slug parsed from the CSV filename (e.g. "heron", "ppdoc_l") — this
        matches the `parsing_model` values in cote_score_cache.parquet. The
        *_gt_predictions.csv duplicate is skipped, since GT comes from
        gt_ssu_bboxes.csv instead.

        gt_df gets an added `ssu_int` column: ssu_id is a string for
        spiritualist (e.g. "ssu_masthead") but an int for hiertext/docbank —
        the mask rasterizer needs a positive int id, so factorize unifies
        both cases.

        page_to_filename maps the cache's page id to the actual image
        filename (e.g. "0001_p001" -> "0001_p001.jpg"). For docbank the
        cache additionally strips the "_ori" suffix that gt_ssu_bboxes.csv
        keeps (e.g. cache page "..._21" vs. filename stem "..._21_ori"), so
        that suffix is stripped here too when building the lookup."""
        gt_csv, pred_dir, _, _, _ = DATASETS[name]
        gt_df = pd.read_csv(REPO_ROOT / gt_csv)
        gt_df["ssu_int"] = pd.factorize(gt_df["ssu_id"])[0] + 1
        page_to_filename = dict(
            zip(
                gt_df["filename"].apply(lambda f: Path(f).stem.removesuffix("_ori")),
                gt_df["filename"],
            )
        )

        pred_parts = []
        for f in sorted((REPO_ROOT / pred_dir).glob(f"{name}_*_predictions.csv")):
            if f.stem.endswith("gt_predictions"):
                continue
            part = pd.read_csv(f)
            # Non-gt prediction CSVs have the full GT box set concatenated in
            # (source == "gt"); render only the real predictions, or the
            # overlay shows an artificial GT-vs-GT overlap on top of the
            # actual predicted boxes.
            if "source" in part.columns:
                part = part[part["source"] != "gt"]
            part["parsing_model"] = f.stem.removeprefix(f"{name}_").removesuffix("_predictions")
            pred_parts.append(part)
        pred_df = pd.concat(pred_parts, ignore_index=True)
        return gt_df, pred_df, page_to_filename

    _cache = {}

    def get_dataset(name):
        if name not in _cache:
            _cache[name] = load_dataset(name)
        return _cache[name]

    return (get_dataset,)


@app.cell
def _(DATASETS, REPO_ROOT, pd):
    def load_quality_table(name):
        """COTe (bbox fast path) joined with CER/SpACER/CDD per OCR model.

        One row per (page, parsing_model, ocr_model): the geometric COTe
        score doesn't depend on ocr_model, so it's repeated across that
        model's rows. The "gt" parsing_model (GT scored against itself) is
        dropped — always near-perfect, not a real model. Sorted worst-SpACER
        first so "high COTe, high SpACER" cases surface early."""
        _, _, _, cote_parquet, cer_parquet = DATASETS[name]
        cote_df = pd.read_parquet(REPO_ROOT / cote_parquet)
        cote_df = cote_df[cote_df["parsing_model"] != "gt"]
        cer_df = pd.read_parquet(REPO_ROOT / cer_parquet)
        cer_df = cer_df[cer_df["parsing_model"] != "gt"]

        merged = cote_df.merge(cer_df, on=["page", "parsing_model"])[
            ["page", "parsing_model", "ocr_model", "cote", "coverage", "overlap",
             "trespass", "excess", "cer", "spacer_total", "cdd_total"]
        ]
        return merged.sort_values("spacer_total", ascending=False).reset_index(drop=True)

    return (load_quality_table,)


@app.cell
def _(
    Image,
    boxes_to_gt_ssu_map,
    boxes_to_pred_masks,
    compute_cote_masks,
    cote_score,
    np,
    plt,
    visualize_cote_states,
):
    def render_page(gt_df, pred_df, image_dir, page, parsing_model, figsize=(8, 10)):
        """Return (original_image_array, overlay_figure) for one (page, model)."""
        gt_boxes_records = gt_df[gt_df["filename"] == page].to_dict("records")
        pred_boxes_records = pred_df[
            (pred_df["filename"] == page) & (pred_df["parsing_model"] == parsing_model)
        ].to_dict("records")

        image_array = np.array(Image.open(image_dir / page).convert("RGB"))
        actual_h, actual_w = image_array.shape[:2]
        csv_w = int(gt_boxes_records[0]["image_width"])
        csv_h = int(gt_boxes_records[0]["image_height"])

        gt_ssu_map = boxes_to_gt_ssu_map(
            gt_boxes_records, csv_w, csv_h, actual_w, actual_h, ssu_id_key="ssu_int"
        )
        pred_masks = boxes_to_pred_masks(pred_boxes_records, csv_w, csv_h, actual_w, actual_h)

        masks = compute_cote_masks(gt_ssu_map, pred_masks)
        cot, cov, ovl, tres, exc = cote_score(gt_ssu_map, pred_masks)

        fig, ax = plt.subplots(figsize=figsize, dpi=100)
        patches = visualize_cote_states(image_array, masks, ax)
        fig.legend(
            handles=patches,
            loc="lower center",
            ncol=max(len(patches), 1),
            framealpha=0.9,
            fontsize=10,
        )
        metrics_text = (
            f"COTe: {cot:.3f}  Coverage: {cov:.3f}  Overlap: {ovl:.3f}  "
            f"Trespass: {tres:.3f}  Excess: {exc:.3f}  "
            f"GT boxes: {len(gt_boxes_records)}  Pred boxes: {len(pred_boxes_records)}"
        )
        fig.suptitle(f"{page} — {parsing_model}\n{metrics_text}", fontsize=11, y=1.02)
        plt.tight_layout()
        plt.close(fig)
        return image_array, fig

    return (render_page,)


@app.cell
def _(mo):
    dataset_picker = mo.ui.dropdown(
        options=["spiritualist", "hiertext", "docbank"],
        value="spiritualist",
        label="Dataset",
    )
    dataset_picker
    return (dataset_picker,)


@app.cell
def _(dataset_picker, get_dataset, load_quality_table):
    gt_df, pred_df, page_to_filename = get_dataset(dataset_picker.value)
    quality_table = load_quality_table(dataset_picker.value)
    return gt_df, page_to_filename, pred_df, quality_table


@app.cell
def _(mo):
    min_cote_slider = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.7, label="Min COTe (parsing looks this good or better)"
    )
    min_cote_slider
    return (min_cote_slider,)


@app.cell
def _(min_cote_slider, quality_table):
    filtered_table = (
        quality_table[quality_table["cote"] >= min_cote_slider.value]
        .sort_values("spacer_total", ascending=False)
        .reset_index(drop=True)
    )
    return (filtered_table,)


@app.cell
def _(filtered_table, mo):
    quality_browser = mo.ui.table(filtered_table, selection=None, page_size=15)
    quality_browser
    return


@app.cell
def _(filtered_table, mo):
    page_picker = mo.ui.dropdown(
        options=list(filtered_table["page"].unique()),
        value=filtered_table["page"].iloc[0],
        label="Page",
        searchable=True,
    )
    page_picker
    return (page_picker,)


@app.cell
def _(filtered_table, mo, page_picker):
    _models = list(
        filtered_table.loc[filtered_table["page"] == page_picker.value, "parsing_model"].unique()
    )
    model_picker = mo.ui.dropdown(options=_models, value=_models[0], label="Parsing model")
    model_picker
    return (model_picker,)


@app.cell
def _(
    DATASETS,
    REPO_ROOT,
    dataset_picker,
    gt_df,
    mo,
    model_picker,
    page_picker,
    page_to_filename,
    pred_df,
    render_page,
):
    _image_dir = REPO_ROOT / DATASETS[dataset_picker.value][2]
    _filename = page_to_filename[page_picker.value]
    _original, _fig = render_page(
        gt_df, pred_df, _image_dir, _filename, model_picker.value
    )
    mo.hstack(
        [mo.image(_original, width=450), mo.as_html(_fig)],
        align="start",
        gap=2,
    )
    return


if __name__ == "__main__":
    app.run()
