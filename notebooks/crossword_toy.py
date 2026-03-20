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
    # SpACER & JSD — Crossword Toy Example

    A minimal worked example using a 6-character crossword:

    - **HAT** across (row 1)
    - **TEAM** down (column 3), sharing **T** with HAT

    ```
    . H A T .
    . . . E .
    . . . A .
    . . . M .
    ```

    Ground truth Q = {A:2, E:1, H:1, M:1, T:1} — 6 characters total.

    Three parsing variants (R_1–R_3) and three OCR-on-GT variants (S_star_1–S_star_3)
    are compared. End-to-end OCR output (S) is available for R_1's boxes only,
    with two different OCR models producing S_1_1 and S_1_2. Runs without S
    show `None` for components that require it.
    """)
    return


@app.cell
def _():
    from collections import Counter
    import matplotlib.pyplot as plt
    import pandas as pd

    from cotescore import cdd_decomp, spacer_decomp, jsd_distance

    # ── Ground truth ────────────────────────────────────────────────────────────
    Q = Counter({"A": 2, "E": 1, "H": 1, "M": 1, "T": 1})

    # ── R: GT characters captured by the parser (per model) ─────────────────────
    # model_1 double-counts T (shared cell hit by two overlapping boxes)
    # model_2 misses H and T entirely
    # model_3 misses H
    R = {
        "R_1": Counter({"A": 2, "E": 1, "H": 1, "M": 1, "T": 2}),
        "R_2": Counter({"A": 1, "E": 1, "M": 1}),
        "R_3": Counter({"A": 1, "E": 1, "M": 1, "T": 1}),
    }

    # ── S*: OCR output when run on perfectly parsed GT regions (per model) ───────
    # model_1: perfect OCR
    # model_2: misses one A, hallucinates B
    # model_3: misses both As, hallucinates B twice
    S_star = {
        "S_star_1": Counter({"A": 2, "E": 1, "H": 1, "M": 1, "T": 1}),
        "S_star_2": Counter({"A": 1, "E": 1, "H": 1, "M": 1, "T": 1, "B": 1}),
        "S_star_3": Counter({"E": 1, "H": 1, "M": 1, "T": 1, "B": 2}),
    }

    # ── S: OCR output on the parsed boxes (model_1 only) ────────────────────────
    # H is lost (OCR fails to read it from the predicted box)
    # T is absent from the predicted boxes
    S = {
        "S_1_1": Counter({"A": 2, "E": 1, "H": 1, "M": 1, "T": 2}),
        "S_1_2": Counter({"A": 2, "E": 1, "H": 1, "M": 1, "T": 1}),
    }

    # ── Runs: (R_key, S_star_key, S_key or None) ────────────────────────────────
    # R_i and S_star_i are paired by index (same parsing scenario).
    # S_1_1 and S_1_2 are two different OCR models run on R_1's parsed boxes.
    RUNS = [
        ("R_1", "S_star_1", "S_1_1"),
        ("R_1", "S_star_1", "S_1_2"),
        ("R_2", "S_star_2", "S_1_1"),
        ("R_3", "S_star_3", "S_1_2"),
    ]
    return (
        Q,
        R,
        RUNS,
        S,
        S_star,
        cdd_decomp,
        jsd_distance,
        pd,
        plt,
        spacer_decomp,
    )


@app.cell
def _(plt):
    """Crossword grid visualisation."""
    _letters = [
        ("H", 1, 1), ("A", 2, 1), ("T", 3, 1),
        ("E", 3, 2), ("A", 3, 3), ("M", 3, 4),
    ]

    _fig, _ax = plt.subplots(figsize=(3, 4))
    for _ch, _x, _y in _letters:
        _ax.add_patch(
            plt.Rectangle(
                (_x - 0.5, -_y + 0.5), 1, 1,
                facecolor="white", edgecolor="black", linewidth=2,
            )
        )
        _ax.text(_x, -_y + 1, _ch, ha="center", va="center", fontsize=22, fontweight="bold")

    _ax.set_xlim(0, 5)
    _ax.set_ylim(-5, 0)
    _ax.set_aspect("equal")
    _ax.axis("off")
    _ax.set_title("Crossword: HAT × TEAM", fontsize=12)
    _fig
    return


@app.cell
def _(Q, R, S, S_star, mo, pd):
    """Distribution counts table — all distributions side by side."""
    _chars = sorted(set().union(Q, *R.values(), *S_star.values(), *S.values()))

    _rows = []
    for _ch in _chars:
        _row = {"char": _ch, "Q": Q.get(_ch, 0)}
        for _k in sorted(R):
            _row[_k] = R[_k].get(_ch, 0)
        for _k in sorted(S_star):
            _row[_k] = S_star[_k].get(_ch, 0)
        for _k in sorted(S):
            _row[_k] = S[_k].get(_ch, 0)
        _rows.append(_row)

    _df = pd.DataFrame(_rows).set_index("char")

    mo.md(
        f"""
    ## Character Count Distributions

    {mo.as_html(_df)}

    **Q** = ground truth &nbsp;|&nbsp;
    **R\\_i** = GT chars captured by parsing variant i &nbsp;|&nbsp;
    **S\\_star\\_i** = OCR on GT regions, variant i &nbsp;|&nbsp;
    **S\\_1\\_1 / S\\_1\\_2** = two OCR models run on R\\_1's parsed boxes
    """
    )
    return


@app.cell
def _(Q, R, RUNS, S, S_star, cdd_decomp, jsd_distance, mo, pd):
    """CDD decomposition (JSD distance) for all runs."""
    _rows = []
    for _r, _ss, _s in RUNS:
        _named = {"gt": Q, "parsing": R[_r], "ocr": S_star[_ss]}
        if _s is not None:
            _named["total"] = S[_s]
        _res = cdd_decomp(_named, metric=jsd_distance)
        _label = f"{_r} / {_ss} / {_s or '—'}"
        _rows.append({
            "run":     _label,
            "d_pars":  round(_res.d_pars,  4) if _res.d_pars  is not None else None,
            "d_ocr":   round(_res.d_ocr,   4) if _res.d_ocr   is not None else None,
            "d_int":   round(_res.d_int,   4) if _res.d_int   is not None else None,
            "d_total": round(_res.d_total, 4) if _res.d_total is not None else None,
        })

    _df = pd.DataFrame(_rows).set_index("run")

    mo.md(
        f"""
    ## CDD Decomposition — JSD Distance

    sqrt-JSD ∈ [0, 1], lower is better. `None` = S not available for that run.

    {mo.as_html(_df)}

    - **d_pars** (JSD(R ∥ Q)): R_1 double-counts T → non-zero despite full coverage.
      R_2/3 miss H and/or T entirely → higher error.
    - **d_ocr** (JSD(S* ∥ Q)): S_star_1 perfect. S_star_2 misses one A, adds B.
      S_star_3 misses both As, adds B twice — highest OCR error.
    - **d_int / d_total**: R_1 runs only. S_1_1 and S_1_2 differ by a hallucinated B
      in S_1_2 → d_ocr and d_total higher for the S_1_2 run.
    """
    )
    return


@app.cell
def _(Q, R, RUNS, S, S_star, mo, pd, spacer_decomp):
    """SpACER decomposition for all runs."""

    def _c2s(counter):
        """Convert a Counter to a plain string (for spacer_decomp)."""
        return "".join(ch * int(v) for ch, v in sorted(counter.items()) if v > 0)

    _rows = []
    for _r, _ss, _s in RUNS:
        _named = {
            "gt":      _c2s(Q),
            "parsing": _c2s(R[_r]),
            "ocr":     _c2s(S_star[_ss]),
        }
        if _s is not None:
            _named["total"] = _c2s(S[_s])
        _res = spacer_decomp(_named)
        _label = f"{_r} / {_ss} / {_s or '—'}"
        _rows.append({
            "run":           _label,
            "d_pars macro":  round(_res.d_pars_macro,  4) if _res.d_pars_macro  is not None else None,
            "d_ocr macro":   round(_res.d_ocr_macro,   4) if _res.d_ocr_macro   is not None else None,
            "d_int macro":   round(_res.d_int_macro,   4) if _res.d_int_macro   is not None else None,
            "d_total macro": round(_res.d_total_macro, 4) if _res.d_total_macro is not None else None,
        })

    _df = pd.DataFrame(_rows).set_index("run")

    mo.md(
        f"""
    ## SpACER Decomposition

    SpACER ∈ [0, ∞), lower is better. `None` = S not available.
    At document level macro = micro (no per-box cancellation possible).

    {mo.as_html(_df)}

    - **d_pars**: R_1 has T×2 against Q's T×1 → Ê > 0 despite full coverage.
      R_2 misses H and T entirely → D > 0, raising the score further.
    - **d_ocr**: S_star_1 = 0 (perfect). S_star_2 adds B and drops one A.
      S_star_3 drops both As and adds B twice — Ê largest here.
    - **d_int / d_total**: R_1 runs only. S_1_2 adds a hallucinated B relative
      to S_1_1 → d_total higher for the S_1_2 run.
    """
    )
    return


@app.cell
def _(Q, R, RUNS, S, S_star, cdd_decomp, jsd_distance, mo, pd, spacer_decomp):
    """Side-by-side JSD vs SpACER comparison for all runs."""

    def _c2s(counter):
        return "".join(ch * int(v) for ch, v in sorted(counter.items()) if v > 0)

    _rows = []
    for _r, _ss, _s in RUNS:
        _named_cdd = {"gt": Q, "parsing": R[_r], "ocr": S_star[_ss]}
        _named_sp  = {"gt": _c2s(Q), "parsing": _c2s(R[_r]), "ocr": _c2s(S_star[_ss])}
        if _s is not None:
            _named_cdd["total"] = S[_s]
            _named_sp["total"]  = _c2s(S[_s])

        _cdd = cdd_decomp(_named_cdd, metric=jsd_distance)
        _sp  = spacer_decomp(_named_sp)
        _label = f"{_r} / {_ss} / {_s or '—'}"

        _rows.append({
            "run":            _label,
            "JSD d_pars":     round(_cdd.d_pars,        4) if _cdd.d_pars        is not None else None,
            "SpACER d_pars":  round(_sp.d_pars_macro,   4) if _sp.d_pars_macro   is not None else None,
            "JSD d_ocr":      round(_cdd.d_ocr,         4) if _cdd.d_ocr         is not None else None,
            "SpACER d_ocr":   round(_sp.d_ocr_macro,    4) if _sp.d_ocr_macro    is not None else None,
            "JSD d_total":    round(_cdd.d_total,       4) if _cdd.d_total       is not None else None,
            "SpACER d_total": round(_sp.d_total_macro,  4) if _sp.d_total_macro  is not None else None,
        })

    _df = pd.DataFrame(_rows).set_index("run")

    mo.md(
        f"""
    ## JSD vs SpACER — Full Comparison

    {mo.as_html(_df)}

    **Key differences:**
    - JSD is bounded at 1; SpACER is not — heavy deletion/insertion pressure
      can push SpACER above 1.
    - JSD is purely distributional: T×2 vs T×1 in R_1 shifts the distribution,
      but a hallucinated B with equal total mass shifts it more. SpACER responds
      directly to the absolute count difference via Ê and penalises deletions via D.
    - S_1_1 vs S_1_2 shows two OCR models on the same parsed boxes: S_1_2
      hallucinates a B, raising both JSD and SpACER d_total while d_pars is identical.
    """
    )
    return


if __name__ == "__main__":
    app.run()
