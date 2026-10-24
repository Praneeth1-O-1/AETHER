"""Analyze AETHER user-study responses exported from Google Forms.

Form question titles must start with their code in brackets, e.g.
"[B1] How well does the predicted land-cover map match ...", matching
questions.csv. Export the responses (Responses -> Download CSV) and run:

    python analyze_user_study.py responses.csv --out results/

Writes summary.json and two charts. Nothing here generates or alters
responses; every number comes from the CSV.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

LIKERT = ["A3", "A4", "B1", "B2", "B3", "B4", "B5", "B6",
          "C1", "C2", "C3", "C4", "C5", "D1", "D2", "D4"]


def load(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    rename = {}
    for col in raw.columns:
        m = re.match(r"\s*\[([A-E]\d+)\]", str(col))
        if m:
            rename[col] = m.group(1)
    df = raw.rename(columns=rename)
    for code in LIKERT:
        if code in df:
            df[code] = pd.to_numeric(df[code].astype(str).str.extract(r"(\d)")[0], errors="coerce")
    return df


def describe(s: pd.Series) -> dict:
    s = s.dropna()
    return {"n": int(s.size), "mean": round(float(s.mean()), 2), "median": float(s.median()),
            "sd": round(float(s.std(ddof=1)), 2) if s.size > 1 else None,
            "pct_agree_4_5": round(float((s >= 4).mean() * 100), 1)}


def cronbach_alpha(items: pd.DataFrame) -> float | None:
    items = items.dropna()
    k = items.shape[1]
    if k < 2 or len(items) < 3:
        return None
    var_sum = items.var(axis=0, ddof=1).sum()
    total_var = items.sum(axis=1).var(ddof=1)
    return round(float(k / (k - 1) * (1 - var_sum / total_var)), 3) if total_var > 0 else None


def wilcoxon(a: pd.Series, b: pd.Series, label: str) -> dict:
    """Paired, non-parametric: Likert data are ordinal, so no t-test."""
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 6:
        return {"comparison": label, "n": len(pair), "note": "too few paired responses"}
    diff = pair.iloc[:, 1] - pair.iloc[:, 0]
    if (diff == 0).all():
        return {"comparison": label, "n": len(pair), "note": "no differences"}
    res = stats.wilcoxon(pair.iloc[:, 0], pair.iloc[:, 1], zero_method="wilcox")
    n_nonzero = int((diff != 0).sum())
    z = stats.norm.isf(res.pvalue / 2) * np.sign(diff.median() or diff.mean())
    return {"comparison": label, "n": len(pair),
            "median_before": float(pair.iloc[:, 0].median()), "median_after": float(pair.iloc[:, 1].median()),
            "W": float(res.statistic), "p": round(float(res.pvalue), 4),
            "effect_r": round(float(abs(z) / np.sqrt(n_nonzero)), 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = load(args.csv)

    clear = df[[c for c in ("B1", "B2", "B3") if c in df]].mean(axis=1)
    cloudy = df[[c for c in ("B4", "B5", "B6") if c in df]].mean(axis=1)
    summary = {
        "participants": int(len(df)),
        "profile": {c: df[c].value_counts().to_dict() for c in ("A1", "A2") if c in df},
        "items": {c: describe(df[c]) for c in LIKERT if c in df},
        "map_quality": {"clear_mean_of_tiles": describe(clear), "cloudy_mean_of_tiles": describe(cloudy)},
        "tests": [wilcoxon(df["C1"], df["C2"], "trust: map only (C1) vs with reliance map (C2)"),
                  wilcoxon(clear, cloudy, "map quality: clear (B1-3) vs cloudy (B4-6)")],
        "reliability_cronbach_alpha": {
            "map_quality_B1_B6": cronbach_alpha(df[[c for c in ("B1", "B2", "B3", "B4", "B5", "B6") if c in df]]),
            "reliance_map_C3_C5": cronbach_alpha(df[[c for c in ("C3", "C4", "C5") if c in df]]),
            "usefulness_D1_D2_D4": cronbach_alpha(df[[c for c in ("D1", "D2", "D4") if c in df]]),
        },
    }
    if "D3" in df:
        uses = df["D3"].dropna().str.split(";|, ", regex=True).explode().str.strip()
        summary["use_cases"] = uses.value_counts().to_dict()
    for c in ("E1", "E2"):
        if c in df:
            summary[f"open_{c}"] = [t for t in df[c].dropna().astype(str) if t.strip()]

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Diverging stacked bars: share of each response level per item.
    items = [c for c in ("B1", "B2", "B3", "B4", "B5", "B6", "C1", "C2", "C3", "C4", "C5", "D1", "D2", "D4") if c in df]
    colors = ["#c8402a", "#e8a493", "#d9d9d4", "#93c7a8", "#2e8a5a"]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(items) + 1), dpi=200)
    for i, c in enumerate(items):
        counts = df[c].value_counts(normalize=True).reindex([1, 2, 3, 4, 5], fill_value=0).values * 100
        left = -(counts[0] + counts[1] + counts[2] / 2)
        for lvl, share in enumerate(counts):
            ax.barh(i, share, left=left, color=colors[lvl], edgecolor="white", linewidth=1)
            left += share
    ax.axvline(0, color="#52514e", lw=0.6)
    ax.set_yticks(range(len(items)), items)
    ax.invert_yaxis()
    ax.set_xlabel("% of participants (left: 1–2, centre: 3, right: 4–5)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "likert.png")

    print(json.dumps({k: summary[k] for k in ("participants", "tests", "reliability_cronbach_alpha")}, indent=2))
    print(f"Wrote {args.out / 'summary.json'} and {args.out / 'likert.png'}")


if __name__ == "__main__":
    main()
