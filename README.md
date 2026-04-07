# SpACER

The Character Error Rate (CER) is a key metric for evaluating the quality of Optical Character Recognition (OCR). However, this metric assumes that text has been perfectly parsed, which is often not the case. Under page-parsing errors, CER becomes undefined, limiting its use as a metric and making evaluating page-level OCR challenging, particularly when using data that do not share a labelling schema. We introduce the Character Error Vector (CEV), a bag-of-characters evaluator for OCR. The CEV can be decomposed into parsing and OCR, and interaction error components. This decomposability allows practitioners to focus on the part of the Document Understanding pipeline that will have the greatest impact on overall text extraction quality. The CEV can be implemented using a variety of methods, of which we demonstrate SpACER (Spatially Aware Character Error Rate) and a Character distribution method using the Jensen-Shannon Distance. We validate the CEV's performance against other metrics: first, the relationship with CER; then, parse quality; and finally, as a direct measure of page-level OCR quality. The validation process shows that the CEV is a valuable bridge between parsing metrics and local metrics like CER. We analyse a dataset of archival newspapers made of degraded images with complex layouts and find that state-of-the-art end-to-end models are outperformed by more traditional pipeline approaches. Whilst the CEV requires character-level positioning for optimal triage, thresholding on easily available values can predict the main error source with an F1 of 0.91. We provide the CEV as part of a Python library to support Document understanding research.

## Companion library

The CEV metrics (SpACER and Jensen-Shannon Distance) described in this paper are implemented in the `cotescore` Python library:

```bash
pip install cotescore
```

See the [cotescore repository](cotescore/README.md) for full documentation and usage examples.

## Repository structure

```
SpACER/
├── notebooks/          # Marimo analysis notebooks (primary entry point)
├── scripts/            # Data preparation, OCR inference, and batch processing
├── ocr_models/         # Wrappers for Tesseract, TrOCR, EasyOCR, and PaddleOCR
└── data/               # Pre-computed results, figures, and case study data
```

### Notebooks

| Notebook | Description |
|---|---|
| `spatial_uncertainty.py` | Validation of SpACER against CER across spatial granularities |
| `ocr_decomposition.py` | CEV decomposition into parsing, OCR, and interaction error components |
| `page_level_validation.py` | Page-level OCR quality validation |
| `end_to_end_analysis.py` | Comparison of end-to-end vs pipeline OCR approaches |
| `char_count_spacer.py` | Character distribution analysis using Jensen-Shannon Distance |
| `spacer_jsd_example.py` | Worked example of SpACER and JSD metrics |
| `analyse_ocr.py` | SSU-level OCR result inspection |
| `visualise_ssu.py` | Structural Semantic Unit visualisation |

### Scripts

| Script | Description |
|---|---|
| `run_ocr.py` / `run_all_ocr.py` | Run OCR models over the dataset |
| `run_bbox_ocr.py` | OCR inference on bounding-box crops |
| `infer_characters.py` | Character-level position inference |
| `extract_ssu_text.py` | Extract text from Structural Semantic Units |
| `page_cer_validation.py` | Compute page-level CER for validation |
| `process_alto_labelled.py` | Process ALTO XML ground-truth files |

## Dataset

The experiments use the **Spiritualist** dataset — a collection of archival newspaper pages with degraded images and complex layouts. Ground-truth SSU bounding boxes can be obtained from xxx

The original images can be obtained from [HuggingFace Spiritualise](https://huggingface.co/datasets/NationalLibraryOfScotland/Spiritualist_Newspaper)

A small **limerick case study** is bundled at `data/limerick_case_study/` and can be used to explore the metrics without needing the full dataset.

## Reproducing the analysis

In order to reproduce the analysis you must first regenerate the OCR using the scripts for each model. Note that Paddle Paddle and pytorch are often incompatible on the same system due to CUDA conflicts and should be run in isolated environments. 

Once the OCR has been regenerated run

Install dependencies using [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Then launch any notebook with [Marimo](https://marimo.io):

```bash
marimo edit notebooks/spatial_uncertainty.py
```

Run the notebooks roughly in this order to follow the paper's validation sequence:

1. `spatial_uncertainty.py` — SpACER vs CER relationship
2. `ocr_decomposition.py` — CEV decomposition
3. `page_level_validation.py` — page-level quality
4. `end_to_end_analysis.py` — end-to-end vs pipeline comparison

## Citation

If you use this work in your research, please cite:

xxxxx

```bibtex
@misc{bourne2026spacer,
  title     = {xxxxx},
  author    = {Bourne, Jonathan and others},
  year      = {2026},
  publisher = {arXiv},
  doi       = {10.48550/arXiv.XXXX.XXXXX},
}
```
