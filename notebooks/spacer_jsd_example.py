import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Parsing Quality — Limerick Case Study

    This notebook measures **d_pars**: how well the predicted bounding boxes
    capture the ground-truth character distribution.

    The limerick dataset provides:
    - **Q** — ground-truth character positions (pixel-level bounding boxes)
    - **Predicted boxes** — 6 bounding box predictions for the page

    From these we build **R** via `build_R`: the GT characters whose midpoints
    fall inside at least one predicted region. R tells us what the parser
    *sees* of the ground truth.

    This is the only component the data supports without actual OCR output.
    Full decomposition (d_ocr, d_int, d_total) requires per-box OCR transcripts.

    > **Data note:** `load_limerick_example()` requires the limerick case study
    > data bundled in the installed `cotescore` package under
    > `cotescore/assets/limerick_case_study/`.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    from cotescore.dataset import load_limerick_example
    from cotescore.adapters import boxes_to_pred_masks
    from cotescore.types import TokenPositions
    from cotescore._distributions import build_Q, build_R
    from cotescore import jsd_distance, spacer

    ground_truth, image, pred_boxes = load_limerick_example()
    img_h, img_w = image.shape[:2]

    print(f"Image:           {img_w}×{img_h}px")
    print(f"Stories:         {len(ground_truth['stories'])}")
    print(f"Prediction boxes: {len(pred_boxes)}")
    return (
        TokenPositions,
        boxes_to_pred_masks,
        build_Q,
        build_R,
        ground_truth,
        image,
        img_h,
        img_w,
        jsd_distance,
        np,
        plt,
        pred_boxes,
        spacer,
    )


@app.cell
def _(TokenPositions, ground_truth, np):
    """Build TokenPositions from GT character bounding boxes."""
    _chars = []
    for _story in ground_truth["stories"].values():
        for _line in _story["lines"]:
            for _ch in _line["characters"]:
                bx, by, bw, bh = _ch["bbox"]
                _chars.append((_ch["char"], int(bx + bw / 2), int(by + bh / 2)))

    token_positions = TokenPositions(
        tokens=np.array([c[0] for c in _chars], dtype=object),
        xs=np.array([c[1] for c in _chars], dtype=int),
        ys=np.array([c[2] for c in _chars], dtype=int),
    )

    print(f"GT characters: {len(token_positions.tokens)} total, "
          f"{len(set(token_positions.tokens.tolist()))} unique")
    return (token_positions,)


@app.cell
def _(boxes_to_pred_masks, img_h, img_w, pred_boxes):
    """Rasterise prediction boxes to binary masks."""
    pred_masks = boxes_to_pred_masks(pred_boxes, img_w, img_h)
    print(f"Masks: {len(pred_masks)}, each {pred_masks[0].shape}")
    return (pred_masks,)


@app.cell
def _(build_Q, build_R, pred_masks, token_positions):
    """Build Q (GT distribution) and R (parsed GT distribution)."""
    Q = build_Q(token_positions)
    R = build_R(token_positions, pred_masks)

    _q_total = sum(Q.values())
    _r_total = sum(R.values())
    print(f"Q total: {_q_total} chars")
    print(f"R total: {_r_total} chars  ({_r_total/_q_total:.1%} of GT — >100% means overlapping boxes double-count)")
    return Q, R


@app.cell
def _(Q, R, plt):
    """Bar chart: Q vs R per character."""
    _chars = sorted(set(Q) | set(R))
    _x = range(len(_chars))
    _q_vals = [Q.get(c, 0) for c in _chars]
    _r_vals = [R.get(c, 0) for c in _chars]

    _fig, _ax = plt.subplots(figsize=(10, 3))
    _w = 0.4
    _ax.bar([i - _w/2 for i in _x], _q_vals, width=_w, label="Q (GT)", color="steelblue")
    _ax.bar([i + _w/2 for i in _x], _r_vals, width=_w, label="R (parsed)", color="coral", alpha=0.85)
    _ax.set_xticks(list(_x))
    _ax.set_xticklabels([repr(c) if c == " " else c for c in _chars])
    _ax.set_ylabel("Count")
    _ax.set_title("Character counts: ground truth Q vs parsed R")
    _ax.legend()
    _fig


@app.cell
def _(Q, R, jsd_distance, mo, spacer):
    """Compute d_pars with both metrics."""
    _jsd = jsd_distance(Q, R)
    _sp  = spacer(Q, R)

    mo.md(
        f"""
    ## d\\_pars — Parsing Quality

    How well do the predicted boxes capture the GT character distribution?

    | Metric | d\\_pars | Range | Interpretation |
    |--------|---------|-------|----------------|
    | JSD distance | `{_jsd:.4f}` | [0, 1] | Distributional shift Q → R |
    | SpACER | `{_sp:.4f}` | [0, ∞) | Count-based error, deletion-sensitive |

    **JSD** measures the shape difference between Q and R as probability
    distributions — insensitive to the total count, only the relative proportions matter.

    **SpACER** responds to absolute count differences (Ê) and net deletions (D).
    Because `build_R` can double-count characters in overlapping boxes,
    R may exceed Q in total count. A SpACER > 0 here reflects the character-level
    count mismatch caused by that overlap, not missing characters.

    Both metrics would be 0 if the predicted boxes perfectly and exclusively
    covered every GT character exactly once.
    """
    )
    return


if __name__ == "__main__":
    app.run()
