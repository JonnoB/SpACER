import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    from collections import Counter
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    from cotescore.ocr import spacer, text_to_counter
    from jiwer import cer

    return Counter, Path, cer, mo, pd, spacer, text_to_counter


@app.cell
def _(mo):
    """Set paths to the GT and OCR text files."""
    gt_path = mo.ui.text(
        label="Ground-truth file",
        value="data/gt.txt",
    )
    ocr_path = mo.ui.text(
        label="OCR file",
        value="data/ocr.txt",
    )

    inputs = mo.hstack([gt_path, ocr_path])
    inputs
    return (gt_path, ocr_path, inputs)


@app.cell
def _(Path, gt_path, mo, ocr_path):
    """Load the two text files."""
    _gt_file = Path(gt_path.value)
    _ocr_file = Path(ocr_path.value)

    if not _gt_file.exists():
        raise FileNotFoundError(f"GT file not found: {_gt_file}")
    if not _ocr_file.exists():
        raise FileNotFoundError(f"OCR file not found: {_ocr_file}")

    gt_text = _gt_file.read_text(encoding="utf-8")
    ocr_text = _ocr_file.read_text(encoding="utf-8")

    mo.md(
        f"**GT length:** {len(gt_text):,} chars  |  "
        f"**OCR length:** {len(ocr_text):,} chars"
    )
    return (gt_text, ocr_text)


@app.cell
def _(Counter, cer, gt_text, ocr_text, pd, spacer, text_to_counter):
    """Compute CER and bag-of-character metrics."""
    # Sequence-based character error rate
    cer_value = float(cer(gt_text, ocr_text))

    # Bag-of-character counters
    gt_counter = text_to_counter(gt_text, mode="char")
    ocr_counter = text_to_counter(ocr_text, mode="char")

    # SpACER — count-based error metric analogous to CER
    spacer_error = spacer(gt_counter, ocr_counter)

    # Bag-of-character accuracy: fraction of GT characters (with multiplicity)
    # correctly present in the OCR output, ignoring order.
    all_chars = set(gt_counter) | set(ocr_counter)
    correct_counts = sum(min(gt_counter[c], ocr_counter[c]) for c in all_chars)
    total_gt_chars = sum(gt_counter.values())
    bag_char_accuracy = (
        correct_counts / total_gt_chars if total_gt_chars > 0 else 0.0
    )

    # Per-character comparison table
    rows = []
    for char in sorted(all_chars):
        gt_count = gt_counter.get(char, 0)
        ocr_count = ocr_counter.get(char, 0)
        rows.append(
            {
                "char": repr(char),
                "gt_count": gt_count,
                "ocr_count": ocr_count,
                "diff": gt_count - ocr_count,
                "correct": min(gt_count, ocr_count),
            }
        )
    comparison_df = pd.DataFrame(rows)

    return (bag_char_accuracy, cer_value, comparison_df, spacer_error)


@app.cell
def _(bag_char_accuracy, cer_value, comparison_df, mo, spacer_error):
    """Display summary and per-character comparison."""
    summary = mo.md(
        f"### GT vs OCR comparison\n\n"
        f"**CER:** `{cer_value:.4f}`\n\n"
        f"**SpACER (bag-of-char error):** `{spacer_error:.4f}`\n\n"
        f"**Bag-of-character accuracy:** `{bag_char_accuracy:.4f}`"
    )

    mo.vstack(
        [
            summary,
            mo.ui.table(comparison_df, label="Per-character counts"),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
