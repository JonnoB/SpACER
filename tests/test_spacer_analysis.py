"""Unit tests for the shared spacer helpers (no dataset files needed)."""

import numpy as np
import pandas as pd
import pytest

from spacer_analysis.detection import coco_match, iou_matrix
from spacer_analysis.names import display_name
from spacer_analysis.paths import DATASETS
from spacer_analysis.tables import bold_best_cols, bold_best_pivot, latex_table
from spacer_analysis.text import normalize_for_cer


class TestNames:
    def test_known_models(self):
        assert display_name("ppdoc_l") == "PPDoc-L"
        assert display_name("PPDOC_L") == "PPDoc-L"
        assert display_name("trocr") == "TrOCR"

    def test_fallback_title_cases(self):
        assert display_name("some_new_model") == "Some-New-Model"


class TestText:
    def test_quotes_dashes_case_whitespace(self):
        assert normalize_for_cer("“Hello” — it’s   FINE") == '"hello" - it\'s fine'

    def test_single_newline_is_space_but_blank_line_kept(self):
        assert normalize_for_cer("a\nb\n\nc") == "a b\n\nc"

    def test_nbsp_and_nfkc(self):
        assert normalize_for_cer("ﬁne\xa0print") == "fine print"


class TestPaths:
    def test_page_id_round_trips(self):
        assert DATASETS["docbank"].page_id("123_ori.jpg") == "123"
        assert DATASETS["docbank"].filename("123") == "123_ori.jpg"
        assert DATASETS["hiertext"].page_id("abc.jpg") == "abc"
        assert DATASETS["spiritualist"].page_id("some/dir/0001_p001.jpg") == "0001_p001"


class TestTables:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({"a": [0.1, 0.3], "b": [0.5, 0.2]},
                            index=pd.Index(["x", "y"], name="parsing_model"))

    def test_bold_best_cols_direction(self, df):
        out = bold_best_cols(df, lower_cols=["a"], higher_cols=["b"])
        assert out.loc["x", "a"] == r"\textbf{0.100}" and out.loc["y", "a"] == "0.300"
        assert out.loc["x", "b"] == r"\textbf{0.500}" and out.loc["y", "b"] == "0.200"

    def test_bold_best_pivot_marks_table_best(self, df):
        out = bold_best_pivot(df, lower_is_better=True)
        assert out.loc["x", "a"] == r"\textbf{0.100}$^{*}$"
        assert out.loc["y", "b"] == r"\textbf{0.200}"
        assert out.loc["x", "b"] == "0.500"

    def test_latex_table_layout(self, df, capsys):
        out = latex_table(df, caption="Cap", label="tab:x")
        assert capsys.readouterr().out.strip() == out.strip()
        lines = [line.strip() for line in out.splitlines()]
        assert lines[0] == r"\begin{table}[htbp]" and lines[1] == r"\centering"
        assert r"\label{tab:x}" in lines and r"\begin{tabular}{lcc}" in lines
        for rule in (r"\toprule", r"\midrule", r"\bottomrule"):
            assert lines.count(rule) == 1
        assert r"\hline" not in out
        # raw index key relabelled, header bolded, floats at 3 dp
        assert r"\textbf{Parsing Model} & \textbf{a} & \textbf{b} \\" in lines
        assert r"x & 0.100 & 0.500 \\" in lines

    def test_latex_table_echo_off(self, df, capsys):
        latex_table(df, caption="Cap", label="tab:x", echo=False)
        assert capsys.readouterr().out == ""

    def test_math_column_label_braces_survive(self, df):
        df = df.rename(columns={"a": r"$\mathbf{R}_\mathbf{ocr}$"})
        out = latex_table(df, caption="Cap", label="tab:x", echo=False)
        assert r"\textbf{$\mathbf{R}_\mathbf{ocr}$}" in out


class TestDetection:
    def test_iou_matrix(self):
        p = np.array([[0, 0, 10, 10], [50, 50, 10, 10]], dtype=float)
        g = np.array([[0, 0, 10, 10], [5, 5, 10, 10]], dtype=float)
        m = iou_matrix(p, g)
        assert m.shape == (2, 2)
        assert m[0, 0] == pytest.approx(1.0)
        assert m[0, 1] == pytest.approx(25 / 175)
        assert m[1].max() == 0.0

    def test_coco_match_confidence_order(self):
        # two predictions both overlap the single GT box; the more confident one wins
        iou = np.array([[0.9], [0.95]])
        assert coco_match(iou, np.array([0.8, 0.2])) == [0.9]
        assert coco_match(iou, np.array([0.2, 0.8])) == [0.95]

    def test_coco_match_takes_next_best_gt(self):
        # pred 0 prefers gt 0; pred 1 also prefers gt 0 but must fall back to gt 1
        iou = np.array([[0.9, 0.0], [0.8, 0.6]])
        assert coco_match(iou, np.array([0.9, 0.5])) == [0.9, 0.6]

    def test_coco_match_threshold(self):
        assert coco_match(np.array([[0.4]]), np.array([1.0])) == []


class TestCorrelation:
    def test_dpars_metric_spearman(self):
        from spacer_analysis.correlation import dpars_metric_spearman

        pages = ["p1", "p2", "p3", "p4"]
        # d_pars identical across OCR models (as in the real data); cote anti-correlated,
        # f1 perfectly correlated, iou constant (-> undefined rho).
        results = pd.DataFrame({
            "page": pages * 2, "parsing_model": "m",
            "ocr_model": ["a"] * 4 + ["b"] * 4,
            "d_pars_spacer_macro": [0.1, 0.2, 0.3, 0.4] * 2,
            "d_pars_cdd": [0.4, 0.3, 0.2, 0.1] * 2,
        })
        cote = pd.DataFrame({"page": pages, "parsing_model": "m", "cote": [0.9, 0.8, 0.7, 0.6]})
        det = pd.DataFrame({"page": pages, "parsing_model": "m",
                            "f1": [0.1, 0.2, 0.3, 0.4], "iou": [0.5] * 4})
        out = dpars_metric_spearman(results, cote, det)
        sp, cdd = out["spacer"].loc["m"], out["cdd"].loc["m"]
        assert sp["COTe"] == pytest.approx(-1.0) and sp["F1@0.5"] == pytest.approx(1.0)
        assert cdd["COTe"] == pytest.approx(1.0) and cdd["F1@0.5"] == pytest.approx(-1.0)
        assert np.isnan(sp["IoU"]) and np.isnan(cdd["IoU"])

    def test_nan_renders_as_dash_and_is_never_best(self):
        df = pd.DataFrame({"a": [np.nan, -0.5]}, index=pd.Index(["x", "y"], name="parsing_model"))
        out = bold_best_cols(df, lower_cols=["a"])
        assert out.loc["x", "a"] == "--" and out.loc["y", "a"] == r"\textbf{-0.500}"
        out = bold_best_pivot(df, lower_is_better=True)
        assert out.loc["x", "a"] == "--" and out.loc["y", "a"] == r"\textbf{-0.500}$^{*}$"
