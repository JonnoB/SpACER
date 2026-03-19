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
    # SpACER and JSD Decomposition — Limerick Case Study

    This notebook demonstrates the four-way decomposition using two metrics:

    - **JSD distance** (sqrt-JSD between character distributions) via `cdd_decomp`
    - **SpACER** (count-based, CER-like) via `spacer_decomp`

    Both operate on the same four named slots:

    | Key | Distribution | Meaning |
    |-----|-------------|---------|
    | `"gt"` | Q | All ground-truth characters |
    | `"parsing"` | R | GT characters spatially captured by predicted regions |
    | `"ocr"` | S* | Characters in GT regions (oracle OCR) |
    | `"total"` | S | Characters in predicted regions (oracle OCR) |

    The **parsing** distribution (R) is built spatially using `build_R`, which
    requires pixel-level character positions — it cannot be derived from text alone.
    This is why `cdd_decomp` accepts pre-built `Counter` objects as well as strings.

    > **Data note:** `load_limerick_example()` requires the limerick case study data
    > to be bundled in the installed `cotescore` package under
    > `cotescore/data/limerick_case_study/`. If that data is absent the cell below
    > will raise a `FileNotFoundError`.
    """)
    return


@app.cell
def _():
    import numpy as np
    from collections import Counter

    from cotescore.dataset import load_limerick_example, extract_ssu_boxes
    from cotescore.adapters import boxes_to_pred_masks, boxes_to_gt_ssu_map
    from cotescore.types import TokenPositions
    from cotescore._distributions import build_Q, build_R, build_S, build_S_star
    from cotescore import (
        jsd_distance,
        cdd_decomp,
        spacer_decomp,
        text_to_counter,
    )

    ground_truth, image, pred_boxes = load_limerick_example()

    img_h, img_w = image.shape[:2]
    print(f"Image: {img_w}×{img_h}px")
    print(f"Stories: {len(ground_truth['stories'])}")
    print(f"Prediction boxes: {len(pred_boxes)}")
    return (
        TokenPositions,
        boxes_to_pred_masks,
        build_Q,
        build_R,
        build_S,
        build_S_star,
        cdd_decomp,
        ground_truth,
        img_h,
        img_w,
        jsd_distance,
        np,
        pred_boxes,
        spacer_decomp,
    )


@app.cell
def _(TokenPositions, ground_truth, np):
    """Build TokenPositions from ground-truth character bounding boxes.

    Each character's pixel midpoint (cx, cy) is used as its spatial position.
    TokenPositions is the structure build_R needs to determine which GT
    characters fall inside each predicted region.
    """
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

    print(f"Total GT characters: {len(token_positions.tokens)}")
    print(f"Unique characters:   {len(set(token_positions.tokens.tolist()))}")
    return (token_positions,)


@app.cell
def _(boxes_to_pred_masks, img_h, img_w, pred_boxes):
    """Rasterise prediction boxes to binary masks (same pixel space as image)."""
    pred_masks = boxes_to_pred_masks(pred_boxes, img_w, img_h)
    print(f"Prediction masks: {len(pred_masks)} masks, each {pred_masks[0].shape}")
    return (pred_masks,)


@app.cell
def _(
    build_Q,
    build_R,
    build_S,
    build_S_star,
    ground_truth,
    pred_masks,
    token_positions,
):
    """Build the four distributions.

    Q  — all GT characters (reference).
    R  — GT characters spatially captured by at least one predicted region.
         Built with build_R, which uses pixel positions — requires Counter input
         to cdd_decomp rather than a string.
    S* — oracle: GT characters grouped by their GT story region.
    S  — oracle: GT characters grouped by predicted region (same source as R,
         different grouping).

    In a real evaluation S* and S would be the actual OCR transcript per region.
    Here we use GT character assignment as an oracle approximation.
    """
    Q = build_Q(token_positions)

    # R requires spatial computation — returns a Counter directly
    R = build_R(token_positions, pred_masks)

    # S*: one token list per GT story (oracle OCR on GT regions)
    gt_token_lists = [
        list("".join(ch["char"] for line in story["lines"] for ch in line["characters"]))
        for story in ground_truth["stories"].values()
    ]
    S_star = build_S_star(gt_token_lists)

    # S: one token list per prediction box (oracle OCR on predicted regions)
    # Derive by assigning each GT character to the first pred box it falls in.
    _pred_char_lists = [[] for _ in pred_masks]
    for _tok, _x, _y in zip(
        token_positions.tokens, token_positions.xs, token_positions.ys
    ):
        for _i, _mask in enumerate(pred_masks):
            if _mask[_y, _x]:
                _pred_char_lists[_i].append(_tok)
                break

    pred_token_lists = [list(chars) for chars in _pred_char_lists]
    S = build_S(pred_token_lists)

    print(f"Q  (GT total):      {sum(Q.values())} tokens, {len(Q)} unique")
    print(f"R  (parsed GT):     {sum(R.values())} tokens, {len(R)} unique")
    print(f"S* (GT-region OCR): {sum(S_star.values())} tokens, {len(S_star)} unique")
    print(f"S  (pred-region OCR): {sum(S.values())} tokens, {len(S)} unique")
    return Q, R, S, S_star, gt_token_lists, pred_token_lists


@app.cell
def _(Q, R, S, S_star, cdd_decomp, jsd_distance, mo):
    """CDD decomposition using JSD distance.

    R is passed as a pre-built Counter because it was constructed spatially
    via build_R — there is no text string that could reconstruct it.
    All other slots could be passed as strings, but Counters work equally well.
    """
    jsd_result = cdd_decomp(
        {"gt": Q, "parsing": R, "ocr": S_star, "total": S},
        metric=jsd_distance,
    )

    mo.md(
        f"""
    ## CDD Decomposition (JSD Distance)

    | Component | Pair | Value |
    |-----------|------|-------|
    | d_pars  | JSD(R ∥ Q)   — parsing captures GT? | `{jsd_result.d_pars:.4f}` |
    | d_ocr   | JSD(S* ∥ Q)  — OCR quality on GT regions | `{jsd_result.d_ocr:.4f}` |
    | d_int   | JSD(S ∥ R)   — interaction error | `{jsd_result.d_int:.4f}` |
    | d_total | JSD(S ∥ Q)   — end-to-end error | `{jsd_result.d_total:.4f}` |

    Values are sqrt-JSD ∈ [0, 1]. Lower is better.
    """
    )
    return (jsd_result,)


@app.cell
def _(gt_token_lists, mo, pred_token_lists, spacer_decomp):
    """SpACER decomposition.

    spacer_decomp takes per-box text lists directly, enabling both macro
    (page-level deletions) and micro (per-box deletions) in one call.
    In oracle mode S == R so d_int ≈ 0.
    """
    # Flatten gt_token_lists to a single string for page-level "gt" slot
    gt_text = "".join("".join(toks) for toks in gt_token_lists)

    # Per-box text strings for pred regions
    pred_texts = ["".join(toks) for toks in pred_token_lists]

    spacer_result = spacer_decomp(
        {
            "gt": gt_text,
            "parsing": pred_texts,   # oracle: GT chars in pred regions
            "ocr": gt_text,          # oracle: OCR on GT regions == GT text
            "total": pred_texts,     # oracle: OCR on pred regions == pred chars
        }
    )

    mo.md(
        f"""
    ## SpACER Decomposition

    Each component is reported at macro (page-level D) and micro (per-box D).
    Micro prevents insertion/deletion cancellation across boxes.

    | Component | Pair | Macro | Micro |
    |-----------|------|-------|-------|
    | d_pars  | SpACER(R, Q) — parsing captures GT? | `{spacer_result.d_pars_macro:.4f}` | `{spacer_result.d_pars_micro:.4f}` |
    | d_ocr   | SpACER(S*, Q) — OCR on GT regions | `{spacer_result.d_ocr_macro:.4f}` | `{spacer_result.d_ocr_micro:.4f}` |
    | d_int   | SpACER(S, R) — interaction | `{spacer_result.d_int_macro:.4f}` | `{spacer_result.d_int_micro:.4f}` |
    | d_total | SpACER(S, Q) — end-to-end | `{spacer_result.d_total_macro:.4f}` | `{spacer_result.d_total_micro:.4f}` |

    SpACER is not bounded at 1 — values > 1 indicate more errors than reference characters.
    Macro and micro diverge when deletions in one box are masked by insertions in another.
    """
    )
    return (spacer_result,)


@app.cell
def _(jsd_result, mo, spacer_result):
    """Side-by-side summary: where the two metrics agree and differ."""
    _d_total_jsd = jsd_result.d_total
    _d_total_spacer_macro = spacer_result.d_total_macro
    _d_total_spacer_micro = spacer_result.d_total_micro

    mo.md(
        f"""
    ## Summary

    | Metric | d_total |
    |--------|---------|
    | JSD distance | `{_d_total_jsd:.4f}` |
    | SpACER macro | `{_d_total_spacer_macro:.4f}` |
    | SpACER micro | `{_d_total_spacer_micro:.4f}` |

    **JSD** captures distributional shift — sensitive to which characters appear
    and in what proportions, regardless of count totals.

    **SpACER macro** blends deletion sensitivity (D) with per-character L1 error
    (Ê). Symmetric around deletions and insertions at page level.

    **SpACER micro** uses per-box deletion counts, so insertions in one box cannot
    hide deletions in another — giving a stricter view of parsing quality.
    """
    )
    return


if __name__ == "__main__":
    app.run()
