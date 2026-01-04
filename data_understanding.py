"""
eda.py

A practical EDA pipeline:
- Data structure overview (dtypes, uniques, memory, duplicates)
- Missing values report + plot
- Numeric & categorical summaries
- Outliers detection (IQR + robust z-score/MAD)
- Correlations + heatmap
- Target relationships + simple hypothesis tests
- Distributions/boxplots/scatters for key columns
All outputs are saved to an output directory.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats


# -----------------------------
# Config
# -----------------------------

@dataclass
class EDAConfig:
    input_path: Path
    output_dir: Path
    target: Optional[str] = None
    time_col: Optional[str] = None
    sample_for_plots: int = 50_000
    max_cols_for_univariate_plots: int = 30
    corr_method: str = "spearman"  # "pearson" or "spearman"
    iqr_k: float = 1.5
    mad_z_threshold: float = 3.5
    random_state: int = 42


# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(p: Path) -> None:
    """Create output directory if needed."""
    p.mkdir(parents=True, exist_ok=True)


def safe_to_datetime(series: pd.Series) -> pd.Series:
    """Try to parse a column to datetime without raising."""
    return pd.to_datetime(series, errors="coerce", utc=False)


def infer_feature_types(df: pd.DataFrame, target: Optional[str], time_col: Optional[str]) -> Dict[str, List[str]]:
    """
    Infer basic feature groups: numeric, categorical, datetime, text/other.
    """
    cols = [c for c in df.columns if c not in {target, time_col}]
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    datetime = [c for c in cols if pd.api.types.is_datetime64_any_dtype(df[c])]
    categorical = [c for c in cols if (pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]))]
    object_cols = [c for c in cols if pd.api.types.is_object_dtype(df[c])]

    # Heuristic: object columns with low unique ratio -> categorical, else text/other
    obj_as_cat = []
    text_other = []
    for c in object_cols:
        nunique = df[c].nunique(dropna=True)
        ratio = nunique / max(len(df), 1)
        if ratio <= 0.05 or nunique <= 50:
            obj_as_cat.append(c)
        else:
            text_other.append(c)

    categorical = sorted(set(categorical + obj_as_cat))
    text_other = sorted(set(text_other))

    return {
        "numeric": sorted(numeric),
        "categorical": categorical,
        "datetime": sorted(datetime),
        "text_other": text_other,
    }


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save dataframe as CSV with safe defaults."""
    df.to_csv(path, index=False)


def plot_missingness(missing: pd.DataFrame, out_path: Path, top_n: int = 30) -> None:
    """Bar plot of missing percentage for top N columns."""
    top = missing.sort_values("missing_pct", ascending=False).head(top_n)
    if top.empty:
        return

    plt.figure()
    plt.bar(top["column"].astype(str), top["missing_pct"].values)
    plt.xticks(rotation=90)
    plt.ylabel("Missing (%)")
    plt.title(f"Top {min(top_n, len(top))} columns by missingness")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def robust_zscore_mad(x: np.ndarray) -> np.ndarray:
    """
    Robust z-score based on MAD (Median Absolute Deviation).
    Reference factor 0.6745 makes it comparable to standard z-score under normality.
    """
    x = x.astype(float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or np.isnan(mad):
        return np.full_like(x, np.nan, dtype=float)
    return 0.6745 * (x - med) / mad


def outliers_report_numeric(df: pd.DataFrame, numeric_cols: List[str], cfg: EDAConfig) -> pd.DataFrame:
    """
    Outliers report using:
    - IQR method (k*IQR)
    - Robust z-score (MAD) threshold
    """
    rows = []
    n = len(df)

    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        x = s.values.astype(float)

        q1 = np.nanpercentile(x, 25) if np.isfinite(x).any() else np.nan
        q3 = np.nanpercentile(x, 75) if np.isfinite(x).any() else np.nan
        iqr = q3 - q1 if np.isfinite(q1) and np.isfinite(q3) else np.nan

        if np.isfinite(iqr) and iqr > 0:
            low = q1 - cfg.iqr_k * iqr
            high = q3 + cfg.iqr_k * iqr
            iqr_out = np.sum((x < low) | (x > high)) if n > 0 else 0
        else:
            low, high, iqr_out = np.nan, np.nan, 0

        rz = robust_zscore_mad(x)
        mad_out = np.sum(np.abs(rz) > cfg.mad_z_threshold) if np.isfinite(rz).any() else 0

        rows.append({
            "column": c,
            "count": n,
            "non_null": int(np.sum(np.isfinite(x))),
            "iqr_q1": q1,
            "iqr_q3": q3,
            "iqr": iqr,
            "iqr_low": low,
            "iqr_high": high,
            "iqr_outliers_count": int(iqr_out),
            "iqr_outliers_pct": (iqr_out / n * 100) if n else 0.0,
            "mad_outliers_count": int(mad_out),
            "mad_outliers_pct": (mad_out / n * 100) if n else 0.0,
        })

    return pd.DataFrame(rows)


def correlation_matrix(df: pd.DataFrame, numeric_cols: List[str], method: str) -> pd.DataFrame:
    """Compute correlation matrix for numeric columns."""
    if not numeric_cols:
        return pd.DataFrame()
    return df[numeric_cols].corr(method=method)


def plot_corr_heatmap(corr: pd.DataFrame, out_path: Path, max_cols: int = 40) -> None:
    """Plot correlation heatmap with matplotlib (no seaborn)."""
    if corr.empty:
        return

    # Keep it readable: limit number of columns
    cols = list(corr.columns)
    if len(cols) > max_cols:
        # Keep highest-variance columns
        variances = corr.var().sort_values(ascending=False)
        cols = list(variances.head(max_cols).index)
        corr = corr.loc[cols, cols]

    data = corr.values
    plt.figure(figsize=(max(6, len(cols) * 0.25), max(6, len(cols) * 0.25)))
    im = plt.imshow(data, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(cols)), cols, rotation=90, fontsize=7)
    plt.yticks(range(len(cols)), cols, fontsize=7)
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def univariate_plots(df: pd.DataFrame, numeric_cols: List[str], out_dir: Path, cfg: EDAConfig) -> None:
    """
    Hist + box for a subset of numeric columns.
    """
    if not numeric_cols:
        return

    # Select subset by missingness ascending (more informative first)
    miss_pct = df[numeric_cols].isna().mean().sort_values()
    cols = list(miss_pct.head(cfg.max_cols_for_univariate_plots).index)

    plot_df = df
    if len(df) > cfg.sample_for_plots:
        plot_df = df.sample(cfg.sample_for_plots, random_state=cfg.random_state)

    for c in cols:
        s = pd.to_numeric(plot_df[c], errors="coerce").dropna()
        if s.empty:
            continue

        # Histogram
        plt.figure()
        plt.hist(s.values, bins=50)
        plt.title(f"Histogram: {c}")
        plt.xlabel(c)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(out_dir / f"hist__{c}.png", dpi=200)
        plt.close()

        # Boxplot
        plt.figure()
        plt.boxplot(s.values, vert=True, showfliers=True)
        plt.title(f"Boxplot: {c}")
        plt.ylabel(c)
        plt.tight_layout()
        plt.savefig(out_dir / f"box__{c}.png", dpi=200)
        plt.close()


def scatter_against_target(df: pd.DataFrame, numeric_cols: List[str], target: str, out_dir: Path, cfg: EDAConfig) -> None:
    """
    Scatter plots for numeric features vs numeric target.
    """
    if target is None or target not in df.columns:
        return
    if not pd.api.types.is_numeric_dtype(df[target]):
        return
    if not numeric_cols:
        return

    plot_df = df[[target] + numeric_cols].copy()
    if len(plot_df) > cfg.sample_for_plots:
        plot_df = plot_df.sample(cfg.sample_for_plots, random_state=cfg.random_state)

    # Choose subset by absolute correlation magnitude
    corr = plot_df.corr(method=cfg.corr_method)[target].drop(index=target).abs().sort_values(ascending=False)
    cols = list(corr.head(min(12, len(corr))).index)

    for c in cols:
        x = pd.to_numeric(plot_df[c], errors="coerce")
        y = pd.to_numeric(plot_df[target], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 50:
            continue

        plt.figure()
        plt.scatter(x[mask].values, y[mask].values, s=8)
        plt.title(f"{c} vs {target}")
        plt.xlabel(c)
        plt.ylabel(target)
        plt.tight_layout()
        plt.savefig(out_dir / f"scatter__{c}__vs__{target}.png", dpi=200)
        plt.close()


def hypothesis_tests(df: pd.DataFrame, feature_groups: Dict[str, List[str]], target: str, cfg: EDAConfig) -> pd.DataFrame:
    """
    Simple hypothesis testing / association measures:
    - Numeric target: Pearson/Spearman for numeric features
    - Binary target: point-biserial for numeric features + chi-square for categorical
    - Multiclass target: ANOVA for numeric + chi-square for categorical
    """
    if target is None or target not in df.columns:
        return pd.DataFrame()

    y = df[target]
    rows = []

    # Determine target type
    y_is_numeric = pd.api.types.is_numeric_dtype(y)
    y_unique = y.dropna().unique()
    y_nunique = len(y_unique)

    # Binary if exactly 2 unique (after dropna)
    y_is_binary = (not y_is_numeric) and (y_nunique == 2)
    if y_is_numeric and y_nunique == 2:
        # Numeric 0/1 also can be treated as binary
        y_is_binary = True

    # Prepare common
    numeric_cols = feature_groups.get("numeric", [])
    cat_cols = feature_groups.get("categorical", [])

    if y_is_numeric and not y_is_binary:
        # Correlations for numeric features
        for c in numeric_cols:
            x = pd.to_numeric(df[c], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() < 30:
                continue

            if cfg.corr_method == "pearson":
                r, p = stats.pearsonr(x[mask].values, y[mask].values)
                metric = "pearson_r"
            else:
                r, p = stats.spearmanr(x[mask].values, y[mask].values)
                metric = "spearman_r"

            rows.append({
                "feature": c,
                "test": metric,
                "statistic": r,
                "p_value": p,
                "n": int(mask.sum()),
                "note": "Association between numeric feature and numeric target",
            })

    elif y_is_binary:
        # Map binary target to 0/1
        y_bin = y.copy()
        if not pd.api.types.is_numeric_dtype(y_bin):
            # Stable mapping
            classes = sorted(pd.Series(y_bin.dropna().unique()).astype(str).tolist())
            mapping = {classes[0]: 0, classes[1]: 1}
            y_bin = y_bin.astype(str).map(mapping)

        y_bin = pd.to_numeric(y_bin, errors="coerce")

        for c in numeric_cols:
            x = pd.to_numeric(df[c], errors="coerce")
            mask = x.notna() & y_bin.notna()
            if mask.sum() < 30:
                continue
            r, p = stats.pointbiserialr(y_bin[mask].values, x[mask].values)
            rows.append({
                "feature": c,
                "test": "point_biserial_r",
                "statistic": r,
                "p_value": p,
                "n": int(mask.sum()),
                "note": "Association between numeric feature and binary target",
            })

        for c in cat_cols:
            ct = pd.crosstab(df[c], y, dropna=True)
            if ct.size == 0 or ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            chi2, p, dof, _ = stats.chi2_contingency(ct.values)
            rows.append({
                "feature": c,
                "test": "chi_square",
                "statistic": chi2,
                "p_value": p,
                "n": int(ct.values.sum()),
                "note": "Independence test for categorical feature vs binary target",
            })

    else:
        # Multiclass categorical target (or any non-numeric with >2 classes)
        # ANOVA for numeric features
        for c in numeric_cols:
            tmp = df[[c, target]].copy()
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
            tmp = tmp.dropna()
            if tmp.empty:
                continue

            groups = [g[c].values for _, g in tmp.groupby(target) if len(g) >= 10]
            if len(groups) < 2:
                continue

            f, p = stats.f_oneway(*groups)
            rows.append({
                "feature": c,
                "test": "anova_f",
                "statistic": f,
                "p_value": p,
                "n": int(len(tmp)),
                "note": "ANOVA across target classes for numeric feature",
            })

        # Chi-square for categorical features
        for c in cat_cols:
            ct = pd.crosstab(df[c], y, dropna=True)
            if ct.size == 0 or ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            chi2, p, dof, _ = stats.chi2_contingency(ct.values)
            rows.append({
                "feature": c,
                "test": "chi_square",
                "statistic": chi2,
                "p_value": p,
                "n": int(ct.values.sum()),
                "note": "Independence test for categorical feature vs multiclass target",
            })

    res = pd.DataFrame(rows)
    if not res.empty:
        res = res.sort_values(["p_value", "feature"], ascending=[True, True])
    return res


# -----------------------------
# Main EDA
# -----------------------------

def run_eda(df: pd.DataFrame, cfg: EDAConfig) -> None:
    ensure_dir(cfg.output_dir)

    # Optional time parsing
    if cfg.time_col and cfg.time_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[cfg.time_col]):
        df[cfg.time_col] = safe_to_datetime(df[cfg.time_col])

    # Overview
    overview = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "non_null": [int(df[c].notna().sum()) for c in df.columns],
        "null": [int(df[c].isna().sum()) for c in df.columns],
        "unique_non_null": [int(df[c].nunique(dropna=True)) for c in df.columns],
        "sample_values": [df[c].dropna().astype(str).head(3).tolist() for c in df.columns],
    })
    overview["null_pct"] = overview["null"] / max(len(df), 1) * 100
    save_csv(overview, cfg.output_dir / "overview__columns.csv")

    # Dataset-level stats
    dataset_stats = pd.DataFrame([{
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "memory_mb_est": float(df.memory_usage(deep=True).sum() / (1024 ** 2)),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_rows_pct": float(df.duplicated().mean() * 100),
    }])
    save_csv(dataset_stats, cfg.output_dir / "overview__dataset.csv")

    # Missingness
    missing = pd.DataFrame({
        "column": df.columns,
        "missing_count": [int(df[c].isna().sum()) for c in df.columns],
    })
    missing["missing_pct"] = missing["missing_count"] / max(len(df), 1) * 100
    save_csv(missing.sort_values("missing_pct", ascending=False), cfg.output_dir / "quality__missingness.csv")
    plot_missingness(missing, cfg.output_dir / "quality__missingness_top.png", top_n=30)

    # Feature groups
    groups = infer_feature_types(df, cfg.target, cfg.time_col)
    pd.DataFrame([{
        "numeric": len(groups["numeric"]),
        "categorical": len(groups["categorical"]),
        "datetime": len(groups["datetime"]),
        "text_other": len(groups["text_other"]),
    }]).to_csv(cfg.output_dir / "overview__feature_groups.csv", index=False)

    # Summaries
    if groups["numeric"]:
        num_desc = df[groups["numeric"]].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
        num_desc.reset_index(names="column", inplace=True)
        save_csv(num_desc, cfg.output_dir / "stats__numeric_describe.csv")

    if groups["categorical"]:
        cat_rows = []
        for c in groups["categorical"]:
            s = df[c].astype("object")
            vc = s.value_counts(dropna=True).head(20)
            cat_rows.append({
                "column": c,
                "unique_non_null": int(s.nunique(dropna=True)),
                "top_values": vc.to_dict(),
            })
        cat_summary = pd.DataFrame(cat_rows)
        save_csv(cat_summary, cfg.output_dir / "stats__categorical_summary.csv")

    # Outliers
    if groups["numeric"]:
        out_rep = outliers_report_numeric(df, groups["numeric"], cfg)
        save_csv(out_rep.sort_values("iqr_outliers_pct", ascending=False), cfg.output_dir / "quality__outliers.csv")

    # Correlations
    if groups["numeric"]:
        corr = correlation_matrix(df, groups["numeric"], cfg.corr_method)
        corr.to_csv(cfg.output_dir / f"relations__corr_{cfg.corr_method}.csv")
        plot_corr_heatmap(corr, cfg.output_dir / f"relations__corr_{cfg.corr_method}_heatmap.png", max_cols=40)

        # Correlation with target if numeric
        if cfg.target and cfg.target in df.columns and pd.api.types.is_numeric_dtype(df[cfg.target]):
            tmp_cols = [cfg.target] + groups["numeric"]
            tmp = df[tmp_cols].copy()
            corr_t = tmp.corr(method=cfg.corr_method)[cfg.target].drop(index=cfg.target).sort_values(key=lambda s: s.abs(), ascending=False)
            corr_t.to_csv(cfg.output_dir / f"relations__corr_with_target_{cfg.target}__{cfg.corr_method}.csv")

    # Univariate plots
    ensure_dir(cfg.output_dir / "plots")
    univariate_plots(df, groups["numeric"], cfg.output_dir / "plots", cfg)

    # Target relationships and hypothesis tests
    if cfg.target and cfg.target in df.columns:
        tests = hypothesis_tests(df, groups, cfg.target, cfg)
        if not tests.empty:
            save_csv(tests, cfg.output_dir / f"relations__hypothesis_tests__target_{cfg.target}.csv")

        scatter_against_target(df, groups["numeric"], cfg.target, cfg.output_dir / "plots", cfg)

    print(f"EDA finished. Outputs saved to: {cfg.output_dir}")


def load_input(path: Path) -> pd.DataFrame:
    """
    Load data from CSV/Parquet.
    Extend here if you need Excel or database sources.
    """
    suffix = path.suffix.lower()
    if suffix in [".csv", ".txt"]:
        return pd.read_csv(path)
    if suffix in [".parquet"]:
        # Parquet requires pyarrow or fastparquet installed
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def parse_args() -> EDAConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str, help="Path to CSV or Parquet")
    parser.add_argument("--outdir", required=True, type=str, help="Output directory for EDA artifacts")
    parser.add_argument("--target", default=None, type=str, help="Target column name (optional)")
    parser.add_argument("--time_col", default=None, type=str, help="Time column name (optional)")
    parser.add_argument("--sample_for_plots", default=50_000, type=int, help="Row sample size for plots")
    parser.add_argument("--corr_method", default="spearman", choices=["pearson", "spearman"])
    args = parser.parse_args()

    return EDAConfig(
        input_path=Path(args.input),
        output_dir=Path(args.outdir),
        target=args.target,
        time_col=args.time_col,
        sample_for_plots=args.sample_for_plots,
        corr_method=args.corr_method,
    )


def main() -> None:
    cfg = parse_args()
    df = load_input(cfg.input_path)
    run_eda(df, cfg)


if __name__ == "__main__":
    main()
