#!/usr/bin/env python3
"""
sensitivity_analysis.py

Sensitivity analyses for the liver-output and interface/context gene-set
definitions
"""

from __future__ import annotations

import gzip
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
Path("/tmp/mouse_crispr_mplconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mouse_crispr_mplconfig")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact
import statsmodels.formula.api as smf


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).parent.resolve()
TABLES_DIR = ROOT / "tables"
SENS_DIR = TABLES_DIR / "sensitivity"
FIGS_DIR = ROOT / "figures"

# Primary analysis outputs from analysis.py
P2_ANNOT = TABLES_DIR / "p2_gene_annotations_v2.csv"
P3_DECOMP = TABLES_DIR / "p3_depleted_decomposition.csv"
P5_COEF = TABLES_DIR / "p5_logistic_coefficients.csv"

# Input data files
NEG_TABLE = ROOT / "mmc4_sup3_negative.csv"
POS_TABLE = ROOT / "mmc4_sup3_positive.csv"
SGRNA_COUNTS_TABLE = ROOT / "SuppTable3_sgRNA_counts.csv"
HUMAN_MOUSE_TABLE = ROOT / "HOM_MouseHumanSequence.rpt"
HOUSEKEEPING_TABLE = ROOT / "Human_Mouse_Common.csv"
DEPMAP_CE_CSV = ROOT / "CRISPRInferredCommonEssentials.csv"

# GO / NCBI files (already downloaded by analysis.py)
GO_BASIC_OBO = ROOT / "go-basic.obo"
NCBI_GENE2GO = ROOT / "gene2go"
NCBI_GENEINFO_GZ = ROOT / "gene_info.gz"

TAXID_MOUSE = 10090

GENE_COL = "id"
FDR_COL = "p.wilcox.bh"
EFFECT_COL = "median.lfc.all"
FDR_THRESHOLD = 0.05
EFFECT_EPS = 0.0

# Primary GO terms copied from analysis.py for documentation
LIVER_OUTPUT_TERMS = {
    "GO:0006805": "xenobiotic metabolic process",
    "GO:0006699": "bile acid biosynthetic process",
    "GO:0006631": "fatty acid metabolic process",
    "GO:0008203": "cholesterol metabolic process",
    "GO:0007596": "blood coagulation",
    "GO:0006956": "complement activation",
    "GO:0009062": "fatty acid catabolic process",
    "GO:0006006": "glucose metabolic process",
}

INTERFACE_PARENT_TERMS = {
    "GO:0019882": "antigen processing and presentation",
    "GO:0015012": "heparan sulfate proteoglycan biosynthetic process",
    "GO:0030203": "glycosaminoglycan metabolic process",
    "GO:0030198": "extracellular matrix organization",
    "GO:0043062": "extracellular structure organization",
    "GO:0007155": "cell adhesion",
    "GO:0098609": "cell-cell adhesion",
    "GO:0007160": "cell-matrix adhesion",
}

# Human Protein Atlas (HPA) focused liver-output gene set.
#
# Rationale: instead of a hand-curated list, the focused liver-output set is
# taken directly from the Human Protein Atlas liver-specific proteome resource.
# We use HPA Table 2: the 12 genes with the highest level of enriched
# expression in liver (Kampf et al., FASEB J 2014 / HPA liver page).
# ALB (albumin) is added as a 13th gene because it is the canonical secreted
# hepatocyte product. Human symbols are mapped to mouse orthologs via MGI
# HOM_MouseHumanSequence.rpt.
#
# Reference (full citation in HPA_REFERENCE):
#   - Human Protein Atlas liver tissue page:
#     https://www.proteinatlas.org/humanproteome/tissue/liver
#   - Uhlén et al. (Science 2015): Tissue-based map of the human proteome.
#     DOI: 10.1126/science.1260419
#   - Fagerberg et al. (Mol Cell Proteomics 2014): genome-wide integration of
#     transcriptomics and antibody-based proteomics.
#     DOI: 10.1074/mcp.M113.035600
#   - Kampf C et al. (FASEB J 2014): The human liver-specific proteome defined
#     by transcriptomics and antibody-based profiling.
#     DOI: 10.1096/fj.14-250555
HPA_TISSUE_FILE = ROOT / "rna_tissue_consensus.tsv"
HPA_LIVER_FC_THRESHOLD = 4.0

HPA_REFERENCE = (
    "Human Protein Atlas. The liver-specific proteome. "
    "URL: https://www.proteinatlas.org/humanproteome/tissue/liver. "
    "See also: Uhlén M et al. Tissue-based map of the human proteome. "
    "Science. 2015;347(6220):1260419. doi:10.1126/science.1260419; "
    "Fagerberg L et al. Analysis of the human tissue-specific expression by "
    "genome-wide integration of transcriptomics and antibody-based proteomics. "
    "Mol Cell Proteomics. 2014;13(2):397-406. doi:10.1074/mcp.M113.035600; "
    "Kampf C et al. The human liver-specific proteome defined by transcriptomics "
    "and antibody-based profiling. FASEB J. 2014;28(9):3903-3914. "
    "doi:10.1096/fj.14-250555"
)

# HPA Table 2: the 12 genes with the highest level of enriched expression in
# liver. Values are the liver nTPM and tissue specificity score (TS) reported
# by HPA (TS = liver nTPM / second-highest tissue nTPM).
# Source: https://www.proteinatlas.org/humanproteome/tissue/liver
HPA_TABLE2_STATS = {
    "SPP2": {"liver_ntpm": 484.9, "ts_score": 4391},
    "AHSG": {"liver_ntpm": 5439.8, "ts_score": 4319},
    "CFHR2": {"liver_ntpm": 1262.2, "ts_score": 2834},
    "F9": {"liver_ntpm": 659.1, "ts_score": 2787},
    "MBL2": {"liver_ntpm": 262.8, "ts_score": 2628},
    "CFHR5": {"liver_ntpm": 156.2, "ts_score": 1562},
    "APOA2": {"liver_ntpm": 33506.8, "ts_score": 1359},
    "SERPINC1": {"liver_ntpm": 4726.5, "ts_score": 1345},
    "F2": {"liver_ntpm": 1138.4, "ts_score": 1206},
    "SLC10A1": {"liver_ntpm": 399.1, "ts_score": 1060},
    "HPX": {"liver_ntpm": 4942.7, "ts_score": 792},
    "CFHR3": {"liver_ntpm": 366.8, "ts_score": 743},
}
HPA_TABLE2_TOP12 = list(HPA_TABLE2_STATS.keys())
# ALB is added as the canonical hepatocyte-output gene.
HPA_CANONICAL_LIVER_OUTPUT = HPA_TABLE2_TOP12 + ["ALB"]


def load_hpa_liver_sets(
    path: Path,
    human_to_mouse: dict[str, str],
    background_genes: set[str],
    fc_threshold: float = HPA_LIVER_FC_THRESHOLD,
) -> tuple[set[str], set[str], dict[str, float]]:
    """
    Return two HPA-derived liver-output sets and human liver fold-changes.

    - tissue_enriched: genes with liver nTPM >= fc_threshold * max other tissue
      (the HPA "tissue enriched" definition).
    - liver_only: genes detected in liver (nTPM >= 1) and not detected in any
      other tissue (the most stringent liver-specific subset).

    Only genes whose mapped mouse ortholog is in background_genes are kept.
    """
    hpa = pd.read_csv(path, sep="\t")
    pt = hpa.pivot_table(index="Gene name", columns="Tissue", values="nTPM", aggfunc="max")
    if "liver" not in pt.columns:
        raise ValueError("'liver' column not found in HPA tissue file")
    liver = pt["liver"]
    other_max = pt.drop(columns="liver").max(axis=1)
    fold = liver / (other_max + 0.1)

    # Tissue-enriched subset
    enriched = fold[fold >= fc_threshold].sort_values(ascending=False)
    tissue_enriched: set[str] = set()
    mouse_fold: dict[str, float] = {}
    for human_symbol, fc in enriched.items():
        mouse_symbol = human_to_mouse.get(human_symbol)
        if mouse_symbol and mouse_symbol in background_genes:
            tissue_enriched.add(mouse_symbol)
            # If multiple human symbols map to the same mouse symbol, keep max FC.
            mouse_fold[mouse_symbol] = max(mouse_fold.get(mouse_symbol, 0.0), fc)

    # Liver-only subset (detected in liver, not detected in any other tissue)
    liver_only_human = set(liver[(liver >= 1) & (other_max < 1)].index)
    liver_only: set[str] = {
        human_to_mouse[h] for h in liver_only_human
        if h in human_to_mouse and human_to_mouse[h] in background_genes
    }

    return tissue_enriched, liver_only, mouse_fold


def build_hpa_table2_set(
    human_to_mouse: dict[str, str],
    background_genes: set[str],
    hpa_path: Path = HPA_TISSUE_FILE,
) -> tuple[set[str], pd.DataFrame]:
    """
    Build the HPA Table 2 focused liver-output set.

    Returns:
        - mouse_symbols: set of mouse orthologs present in the screen.
        - stats_df: DataFrame with human_symbol, hpa_liver_ntpm, ts_score,
          mouse_symbol, and a note for genes without a mouse ortholog.
    """
    # Compute ALB stats from the local HPA file (ALB is not in the top 12).
    hpa = pd.read_csv(hpa_path, sep="\t")
    pt = hpa.pivot_table(index="Gene name", columns="Tissue", values="nTPM", aggfunc="max")
    liver = pt["liver"]
    other_max = pt.drop(columns="liver").max(axis=1)
    alb_ntpm = liver.get("ALB", float("nan"))
    alb_ts = (
        round(alb_ntpm / (other_max.get("ALB", 0.0) + 0.1), 1)
        if "ALB" in liver.index
        else float("nan")
    )

    stats_rows = []
    mouse_symbols: set[str] = set()
    for human_symbol in HPA_CANONICAL_LIVER_OUTPUT:
        if human_symbol in HPA_TABLE2_STATS:
            ntpm = HPA_TABLE2_STATS[human_symbol]["liver_ntpm"]
            ts = HPA_TABLE2_STATS[human_symbol]["ts_score"]
        else:
            ntpm = alb_ntpm
            ts = alb_ts
        mouse_symbol = human_to_mouse.get(human_symbol)
        if mouse_symbol and mouse_symbol in background_genes:
            mouse_symbols.add(mouse_symbol)
            note = ""
        else:
            note = "no mouse ortholog in screen" if not mouse_symbol else "mouse ortholog not in screen"
        stats_rows.append({
            "human_symbol": human_symbol,
            "hpa_liver_ntpm": ntpm,
            "hpa_tissue_specificity_score": ts,
            "mouse_symbol": mouse_symbol if mouse_symbol else "",
            "in_screen": int(mouse_symbol in background_genes) if mouse_symbol else 0,
            "note": note,
        })
    return mouse_symbols, pd.DataFrame(stats_rows)


# ============================================================
# UTILITIES
# ============================================================


def ensure_dirs() -> None:
    SENS_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in (GENE_COL, FDR_COL, EFFECT_COL):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}")
    return df


def extract_technical_covariates(counts_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(counts_csv)
    cov = (
        df.groupby("gene")
        .agg(
            sgrna_count=("sgRNA", "count"),
            baseline_abundance=("plasmid.ttgact", "median"),
        )
        .reset_index()
    )
    cov = cov.rename(columns={"gene": GENE_COL})
    cov["log2_abundance"] = np.log2(cov["baseline_abundance"] + 1)
    return cov


def build_human_to_mouse(path: Path) -> dict[str, str]:
    """Parse MGI HOM_MouseHumanSequence.rpt to map human symbol -> mouse symbol."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    key_col, taxon_col, symbol_col = "DB Class Key", "NCBI Taxon ID", "Symbol"
    mouse = (
        df[df[taxon_col] == "10090"][[key_col, symbol_col]]
        .rename(columns={symbol_col: "mouse_symbol"})
        .dropna()
    )
    human = (
        df[df[taxon_col] == "9606"][[key_col, symbol_col]]
        .rename(columns={symbol_col: "human_symbol"})
        .dropna()
    )
    merged = human.merge(mouse, on=key_col, how="inner")
    merged = merged.drop_duplicates(subset=["human_symbol"])
    return dict(
        zip(
            merged["human_symbol"].str.strip(),
            merged["mouse_symbol"].str.strip(),
        )
    )


def build_symbol2geneid(gene_info_gz: Path, taxid: int) -> dict[str, int]:
    """Map mouse symbols (and unambiguous synonyms) to NCBI Gene IDs."""
    official: dict[str, int] = {}
    syn_to_gids: defaultdict[str, set[int]] = defaultdict(set)
    with gzip.open(gene_info_gz, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5 or parts[0] != str(taxid):
                continue
            gid = int(parts[1])
            sym = parts[2]
            if sym and sym != "-":
                official[sym] = gid
            for syn in parts[4].split("|"):
                if syn and syn != "-":
                    syn_to_gids[syn].add(gid)
    result = dict(official)
    for syn, gids in syn_to_gids.items():
        if syn not in result and len(gids) == 1:
            result[syn] = next(iter(gids))
    return result


def symbols_to_geneids(symbols: set[str], sym2id: dict[str, int]) -> set[int]:
    return {sym2id[s] for s in symbols if s in sym2id}


def load_housekeeping_symbols(path: Path) -> set[str]:
    """Read housekeeping gene symbols from the unicamp CSV."""
    # File has a single header "Mouse;Human" separated by semicolons.
    df = pd.read_csv(path, sep=None, engine="python")
    if len(df.columns) == 1 and ";" in str(df.columns[0]):
        df = pd.read_csv(path, sep=";")
    norm_to_orig = {c.lower().replace(" ", "_"): c for c in df.columns}
    preferred = [
        "mouse",
        "mouse_gene_symbol",
        "mouse_symbol",
        "mgi_gene_symbol",
        "mgi_symbol",
        "gene_symbol_mouse",
        "mouse_gene",
        "gene_symbol",
        "symbol",
        "gene",
    ]
    chosen = None
    for key in preferred:
        if key in norm_to_orig:
            chosen = norm_to_orig[key]
            break
    if chosen is None:
        raise ValueError(f"Could not find a mouse symbol column in {path}; columns={df.columns.tolist()}")
    vals = df[chosen].dropna().astype(str)
    return {v.strip() for v in vals if v.strip() not in {"-", "NA", "NaN", "nan"}}


def load_depmap_common_essentials(path: Path, human_to_mouse: dict[str, str]) -> set[str]:
    """Return mouse symbols for human DepMap common essentials."""
    df = pd.read_csv(path, header=None, names=["entry"])
    human_symbols = {str(e).split("(")[0].strip() for e in df["entry"].dropna()}
    return {human_to_mouse[h] for h in human_symbols if h in human_to_mouse}


def parse_obo(path: Path) -> dict[str, dict]:
    """Minimal OBO parser returning id -> {name, namespace, parents, children}."""
    terms: dict[str, dict] = {}
    current: Optional[dict] = None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line == "[Term]":
                current = {"parents": [], "children": []}
            elif line == "":
                if current and current.get("id"):
                    terms[current["id"]] = current
                current = None
            elif current is not None:
                if line.startswith("id: "):
                    current["id"] = line[4:]
                elif line.startswith("name: "):
                    current["name"] = line[6:]
                elif line.startswith("namespace: "):
                    current["namespace"] = line[11:]
                elif line.startswith("is_a: "):
                    parent = line[6:].split(" ! ")[0]
                    current["parents"].append(parent)
    # Build children
    for tid, term in terms.items():
        for p in term["parents"]:
            if p in terms:
                terms[p]["children"].append(tid)
    return terms


def get_descendants(terms: dict[str, dict], root_id: str) -> set[str]:
    """Return all descendant term IDs (including root)."""
    out: set[str] = {root_id}
    stack = list(terms.get(root_id, {}).get("children", []))
    while stack:
        tid = stack.pop()
        if tid not in out:
            out.add(tid)
            stack.extend(terms.get(tid, {}).get("children", []))
    return out


def load_gene2go_cc(path: Path, taxid: int) -> dict[int, set[str]]:
    """Read NCBI gene2go and return gene_id -> set of CC GO terms."""
    out: dict[int, set[str]] = defaultdict(set)
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8 or parts[0] != str(taxid):
                continue
            gid = int(parts[1])
            go_id = parts[2]
            category = parts[7]  # Category column
            if category == "Component":
                out[gid].add(go_id)
    return dict(out)


def fisher_exact_with_ci(
    study: set[int], term_genes: set[int], universe: set[int], alternative: str = "less"
) -> dict:
    a = len(study & term_genes)
    b = len(study - term_genes)
    c = len((universe - study) & term_genes)
    d = len((universe - study) - term_genes)
    odds, p = fisher_exact([[a, b], [c, d]], alternative=alternative)
    # Log CI for odds ratio
    if a == 0 or b == 0 or c == 0 or d == 0:
        # Add 0.5 to all cells for CI calculation
        a_, b_, c_, d_ = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    else:
        a_, b_, c_, d_ = a, b, c, d
    log_or = math.log((a_ * d_) / (b_ * c_))
    se_log_or = math.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
    ci_lo = math.exp(log_or - 1.96 * se_log_or)
    ci_hi = math.exp(log_or + 1.96 * se_log_or)
    return {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "odds_ratio": odds,
        "p": p,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "study_size": len(study),
        "term_size": len(term_genes),
        "universe_size": len(universe),
    }


def fit_logistic_predictor(
    df_model: pd.DataFrame,
    predictor: str,
    covariates: list[str] = None,
) -> dict:
    """
    Fit logistic regression is_depleted ~ predictor + covariates.
    Return coefficient, OR, CI, p, AIC.
    """
    covariates = covariates or ["sgrna_count", "log2_abundance"]
    cols = ["is_depleted", predictor] + covariates
    df = df_model[cols].dropna()
    if df["is_depleted"].sum() < 10 or df[predictor].sum() == 0:
        return {"coef": np.nan, "or": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan, "aic": np.nan}
    for col in covariates:
        sd = df[col].std()
        if sd > 0:
            df = df.copy()
            df[col] = (df[col] - df[col].mean()) / sd
    formula = f"is_depleted ~ {' + '.join([predictor] + covariates)}"
    try:
        m = smf.logit(formula=formula, data=df).fit(disp=0)
        b = m.params[predictor]
        se = m.bse[predictor]
        p = m.pvalues[predictor]
        or_ = math.exp(b)
        ci_lo = math.exp(b - 1.96 * se)
        ci_hi = math.exp(b + 1.96 * se)
        return {"coef": b, "or": or_, "ci_lo": ci_lo, "ci_hi": ci_hi, "p": p, "aic": m.aic}
    except Exception as exc:
        print(f"    [!] Logistic failed for {predictor}: {exc}")
        return {"coef": np.nan, "or": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan, "aic": np.nan}


def pseudo_r2(model) -> float:
    """Nagelkerke pseudo-R2 for a fitted statsmodels LogitResults."""
    if model is None:
        return np.nan
    ll0 = model.llnull
    llf = model.llf
    n = model.nobs
    if ll0 == 0 or n == 0:
        return np.nan
    cox = 1 - math.exp(2 * (ll0 - llf) / n)
    return cox / (1 - math.exp(2 * ll0 / n))


def plot_model_sensitivity_forest(s5: pd.DataFrame, out_file: Path) -> None:
    """Plot the DepMap and housekeeping common-essential sensitivity models."""
    primary = s5[
        (s5["lo_set"] == "primary_liver_output")
        & (s5["if_set"] == "primary_interface_context")
        & (s5["ce_set"].isin(["depmap_common_essential", "housekeeping"]))
    ].copy()
    if primary.empty:
        print("  [!] No primary sensitivity rows available for forest plot")
        return

    model_labels = {
        "depmap_common_essential": "DepMap common essential",
        "housekeeping": "UNICAMP housekeeping",
    }
    term_specs = [
        ("ce", {"depmap_common_essential": "Common essential", "housekeeping": "Housekeeping"}),
        ("lo", {"depmap_common_essential": "Liver-output GO", "housekeeping": "Liver-output GO"}),
        ("if", {"depmap_common_essential": "Interface/context", "housekeeping": "Interface/context"}),
    ]

    rows = []
    for ce_set in ["depmap_common_essential", "housekeeping"]:
        match = primary[primary["ce_set"] == ce_set]
        if match.empty:
            continue
        row = match.iloc[0]
        for prefix, labels in term_specs:
            rows.append({
                "model": model_labels[ce_set],
                "label": labels[ce_set],
                "term": prefix,
                "or": row[f"{prefix}_or"],
                "ci_lo": row[f"{prefix}_ci_lo"],
                "ci_hi": row[f"{prefix}_ci_hi"],
                "p": row[f"{prefix}_p"],
            })

    plot_df = pd.DataFrame(rows)
    plot_df["y"] = list(range(len(plot_df)))[::-1]

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    colors = {"ce": "#464646", "lo": "#f0c864", "if": "#a3b4ff"}
    for _, row in plot_df.iterrows():
        color = colors[row["term"]]
        ax.plot(
            [row["ci_lo"], row["ci_hi"]],
            [row["y"], row["y"]],
            color=color,
            linewidth=2.0,
            solid_capstyle="round",
            zorder=3,
        )
        ax.scatter(row["or"], row["y"], color=color, s=48, zorder=4)
        p_str = f"p={row['p']:.1e}" if row["p"] < 0.01 else f"p={row['p']:.3f}"
        ax.text(105, row["y"], p_str, va="center", fontsize=7.5)

    labels = [f"{row.model}: {row.label}" for row in plot_df.itertuples()]
    ax.set_xscale("log")
    ax.set_xlim(0.04, 220)
    ax.axvline(1.0, color="0.55", linestyle="--", linewidth=0.9, zorder=1)
    ax.set_yticks(plot_df["y"])
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Odds ratio for depletion (95% CI)", fontsize=9)
    ax.set_title("Full-model sensitivity to common-essential definition", fontsize=10, pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_file}")


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    ensure_dirs()
    print("=" * 60)
    print("Sensitivity analysis for liver-output / interface gene sets")
    print("=" * 60)

    # ---- Load primary screen data ---------------------------------
    neg = load_csv(NEG_TABLE)
    pos = load_csv(POS_TABLE)
    neg[GENE_COL] = neg[GENE_COL].astype(str)
    pos[GENE_COL] = pos[GENE_COL].astype(str)

    cov_df = extract_technical_covariates(SGRNA_COUNTS_TABLE)
    neg = neg.merge(cov_df, on=GENE_COL, how="left")

    background_genes = set(neg[GENE_COL].unique())
    sig_neg = neg.loc[neg[FDR_COL] < FDR_THRESHOLD]
    sig_pos = pos.loc[pos[FDR_COL] < FDR_THRESHOLD]
    depleted = set(sig_neg.loc[sig_neg[EFFECT_COL] < -EFFECT_EPS, GENE_COL].unique())
    enriched = set(sig_pos.loc[sig_pos[EFFECT_COL] > EFFECT_EPS, GENE_COL].unique())

    print(f"  Universe: {len(background_genes)}")
    print(f"  Depleted: {len(depleted)}")
    print(f"  Enriched: {len(enriched)}")

    # ---- Load primary annotations from analysis.py --------------
    annot = pd.read_csv(P2_ANNOT)
    annot["gene"] = annot["gene"].astype(str)
    annot = annot.set_index("gene")

    primary_ce = set(annot.index[annot["is_common_essential"] == 1])
    primary_lo = set(annot.index[annot["is_liver_output_go"] == 1])
    primary_if = set(annot.index[annot["is_interface"] == 1])
    primary_if_subcat = (
        annot.loc[annot["is_interface"] == 1, "interface_subcat"]
        .dropna()
        .to_dict()
    )

    print(f"  Primary common essential: {len(primary_ce)}")
    print(f"  Primary liver-output:     {len(primary_lo)}")
    print(f"  Primary interface:        {len(primary_if)}")

    # ---- Build symbol -> geneid mapping ---------------------------
    sym2id = build_symbol2geneid(NCBI_GENEINFO_GZ, TAXID_MOUSE)
    id2sym = {v: k for k, v in sym2id.items()}

    universe_ids = symbols_to_geneids(background_genes, sym2id)
    depleted_ids = symbols_to_geneids(depleted, sym2id)
    universe_set = set(universe_ids)
    depleted_set = set(depleted_ids)

    screen_sym_to_id = {g: sym2id[g] for g in background_genes if g in sym2id}

    # ---- Human -> mouse mapping for external data -----------------
    human_to_mouse = build_human_to_mouse(HUMAN_MOUSE_TABLE)
    print(f"  Human->mouse ortholog map: {len(human_to_mouse)} pairs")

    # ---- Build alternative gene sets ------------------------------
    alternative_sets: dict[str, dict] = {}

    # HPA Table 2 focused liver-output set (replaces hand-curated canonical set)
    hpa_table2_symbols, hpa_table2_stats = build_hpa_table2_set(
        human_to_mouse, background_genes, HPA_TISSUE_FILE
    )
    alternative_sets["canonical_liver_output"] = {
        "symbols": hpa_table2_symbols,
        "ids": symbols_to_geneids(hpa_table2_symbols, sym2id),
        "source": (
            "Human Protein Atlas Table 2: top 12 liver-enriched genes plus ALB "
            f"({len(hpa_table2_symbols)} of {len(HPA_CANONICAL_LIVER_OUTPUT)} human genes map to mouse orthologs in screen)"
        ),
        "category": "liver_output",
    }

    # Strict HPA liver-only subset for sensitivity check
    _, hpa_liver_only_symbols, _ = load_hpa_liver_sets(
        HPA_TISSUE_FILE, human_to_mouse, background_genes, HPA_LIVER_FC_THRESHOLD
    )
    alternative_sets["hpa_liver_only"] = {
        "symbols": hpa_liver_only_symbols,
        "ids": symbols_to_geneids(hpa_liver_only_symbols, sym2id),
        "source": (
            f"Human Protein Atlas liver-only detected genes (detected in liver and "
            f"not detected in any other tissue; {len(hpa_liver_only_symbols)} mouse orthologs in screen)"
        ),
        "category": "liver_output",
    }
    print(f"  HPA Table 2 focused set (mouse orthologs in screen): {len(hpa_table2_symbols)}")
    print(f"  HPA liver-only detected (mouse orthologs in screen): {len(hpa_liver_only_symbols)}")

    # ---- Common-essential alternatives -----------------------------
    hk_symbols = load_housekeeping_symbols(HOUSEKEEPING_TABLE)
    hk_symbols &= background_genes
    ce_symbols = load_depmap_common_essentials(DEPMAP_CE_CSV, human_to_mouse)
    ce_symbols &= background_genes
    common_essential_alternatives = {
        "depmap_common_essential": ce_symbols,
        "housekeeping": hk_symbols,
    }
    print(f"  DepMap CE: {len(ce_symbols)}")
    print(f"  Housekeeping: {len(hk_symbols)}")

    # ============================================================
    # STEP 1: Document primary sets
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 1 — Documenting primary gene sets")
    print("=" * 60)

    primary_definitions = []
    for name, symbols, source in [
        ("common_essential", primary_ce, "DepMap CRISPRInferredCommonEssentials via MGI homology"),
        ("liver_output", primary_lo, "analysis.py LIVER_OUTPUT_TERMS (8 GO BP terms + descendants)"),
        ("interface_context", primary_if, "analysis.py INTERFACE_GO_TERMS (8 parent GO BP terms + descendants)"),
    ]:
        ids = symbols_to_geneids(symbols, sym2id)
        dep = symbols & depleted
        dep_ids = ids & depleted_set
        primary_definitions.append({
            "set_name": name,
            "source": source,
            "n_genes": len(symbols),
            "n_gene_ids": len(ids),
            "n_depleted": len(dep),
            "depletion_rate": len(dep) / len(symbols) if symbols else 0.0,
        })
    pd.DataFrame(primary_definitions).to_csv(
        SENS_DIR / "s1_primary_gene_set_definitions.csv", index=False
    )
    print(f"  Saved: {SENS_DIR / 's1_primary_gene_set_definitions.csv'}")

    # Primary gene lists
    rows = []
    for gene in sorted(background_genes):
        if gene not in annot.index:
            continue
        cats = []
        if gene in primary_ce:
            cats.append("common_essential")
        if gene in primary_lo:
            cats.append("liver_output")
        if gene in primary_if:
            cats.append("interface_context")
        if not cats:
            continue
        for cat in cats:
            rows.append({
                "gene": gene,
                "set_name": cat,
                "subcategory": primary_if_subcat.get(gene, ""),
                "median_lfc": annot.loc[gene, EFFECT_COL],
                "fdr": annot.loc[gene, FDR_COL],
                "is_depleted": int(gene in depleted),
            })
    pd.DataFrame(rows).to_csv(SENS_DIR / "s2_primary_gene_lists.csv", index=False)
    print(f"  Saved: {SENS_DIR / 's2_primary_gene_lists.csv'}")

    # Overlap matrix (Jaccard)
    primary_sets = {
        "common_essential": primary_ce,
        "liver_output": primary_lo,
        "interface_context": primary_if,
    }
    overlap_rows = []
    for n1, s1 in primary_sets.items():
        for n2, s2 in primary_sets.items():
            inter = len(s1 & s2)
            union = len(s1 | s2)
            overlap_rows.append({
                "set_a": n1,
                "set_b": n2,
                "intersection": inter,
                "union": union,
                "jaccard": inter / union if union else 0.0,
            })
    pd.DataFrame(overlap_rows).to_csv(SENS_DIR / "s3_set_overlap_matrix.csv", index=False)
    print(f"  Saved: {SENS_DIR / 's3_set_overlap_matrix.csv'}")

    # ============================================================
    # STEP 2: Test alternative sets
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 2 — Testing alternative gene sets")
    print("=" * 60)

    df_model = neg[[GENE_COL, "sgrna_count", "log2_abundance"]].copy()
    df_model = df_model.rename(columns={GENE_COL: "gene"})
    df_model["is_depleted"] = df_model["gene"].isin(depleted).astype(int)
    df_model["entrez_id"] = df_model["gene"].map(sym2id)
    df_model = df_model.dropna(subset=["entrez_id", "sgrna_count", "log2_abundance"])

    test_rows = []
    all_sets = {
        "primary_liver_output": {"symbols": primary_lo, "ids": symbols_to_geneids(primary_lo, sym2id), "source": "Primary GO-based", "category": "liver_output"},
        "primary_interface_context": {"symbols": primary_if, "ids": symbols_to_geneids(primary_if, sym2id), "source": "Primary GO-based", "category": "interface"},
        **alternative_sets,
    }

    for set_name, info in all_sets.items():
        symbols = info["symbols"]
        ids = info["ids"]
        dep_symbols = symbols & depleted
        dep_ids = ids & depleted_set
        category = info["category"]

        # Fisher exact test
        alt = "less" if category == "liver_output" else "greater"
        fisher = fisher_exact_with_ci(depleted_set, ids, universe_set, alternative=alt)

        # Logistic regression
        predictor = f"is_{set_name}"
        df_model[predictor] = df_model["entrez_id"].isin(ids).astype(int)
        lr = fit_logistic_predictor(df_model, predictor)

        test_rows.append({
            "set_name": set_name,
            "category": category,
            "source": info["source"],
            "n_genes": len(symbols),
            "n_depleted": len(dep_symbols),
            "depletion_rate": len(dep_symbols) / len(symbols) if symbols else 0.0,
            "fisher_or": fisher["odds_ratio"],
            "fisher_p": fisher["p"],
            "fisher_ci_lo": fisher["ci_lo"],
            "fisher_ci_hi": fisher["ci_hi"],
            "logistic_or": lr["or"],
            "logistic_p": lr["p"],
            "logistic_ci_lo": lr["ci_lo"],
            "logistic_ci_hi": lr["ci_hi"],
            "logistic_aic": lr["aic"],
        })

    s4 = pd.DataFrame(test_rows)
    s4.to_csv(SENS_DIR / "s4_alternative_set_enrichment.csv", index=False)
    print(f"  Saved: {SENS_DIR / 's4_alternative_set_enrichment.csv'}")
    print("\n  Alternative set results:")
    print(s4[["set_name", "n_genes", "n_depleted", "depletion_rate", "fisher_or", "logistic_or", "logistic_p"]].to_string(index=False))

    # ============================================================
    # STEP 3: Full model sensitivity
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 3 — Full logistic model sensitivity")
    print("=" * 60)

    model_rows = []
    base_formula_terms = ["sgrna_count", "log2_abundance"]

    # Standardise covariates once
    df_std = df_model.copy()
    for col in base_formula_terms:
        sd = df_std[col].std()
        if sd > 0:
            df_std[col] = (df_std[col] - df_std[col].mean()) / sd

    def run_full_model(df: pd.DataFrame, lo_col: str, if_col: str, ce_col: str = "is_common_essential") -> dict:
        required = ["is_depleted", lo_col, if_col, ce_col] + base_formula_terms
        df_sub = df[required].dropna()
        formula = f"is_depleted ~ {ce_col} + {lo_col} + {if_col} + sgrna_count + log2_abundance"
        try:
            m = smf.logit(formula=formula, data=df_sub).fit(disp=0)
            r2 = pseudo_r2(m)
            out: dict = {"aic": m.aic, "pseudo_r2": r2, "nobs": int(m.nobs)}
            term_map = {ce_col: "ce", lo_col: "lo", if_col: "if"}
            for term, prefix in term_map.items():
                b = m.params[term]
                se = m.bse[term]
                p = m.pvalues[term]
                out[f"{prefix}_or"] = math.exp(b)
                out[f"{prefix}_ci_lo"] = math.exp(b - 1.96 * se)
                out[f"{prefix}_ci_hi"] = math.exp(b + 1.96 * se)
                out[f"{prefix}_p"] = p
            return out
        except Exception as exc:
            print(f"    [!] Full model failed for {lo_col}/{if_col}: {exc}")
            return {}

    # Primary model and common-essential definitions
    df_std["is_common_essential"] = df_std["entrez_id"].isin(symbols_to_geneids(primary_ce, sym2id)).astype(int)
    df_std["is_primary_liver_output"] = df_std["entrez_id"].isin(symbols_to_geneids(primary_lo, sym2id)).astype(int)
    df_std["is_primary_interface_context"] = df_std["entrez_id"].isin(symbols_to_geneids(primary_if, sym2id)).astype(int)

    # Sensitivity models: replace one predictor at a time
    # Skip very small alternative sets in the full model (unstable estimates).
    for set_name, info in alternative_sets.items():
        ids = info["ids"]
        if len(ids) < 50:
            print(f"    [skip] Full-model sensitivity for {set_name} (n={len(ids)} < 50)")
            continue
        col = f"is_{set_name}"
        df_std[col] = df_std["entrez_id"].isin(ids).astype(int)
        if info["category"] == "liver_output":
            res = run_full_model(df_std, col, "is_primary_interface_context")
            if res:
                model_rows.append({"lo_set": set_name, "if_set": "primary_interface_context", "ce_set": "depmap_common_essential", **res})
        elif info["category"] == "interface":
            res = run_full_model(df_std, "is_primary_liver_output", col)
            if res:
                model_rows.append({"lo_set": "primary_liver_output", "if_set": set_name, "ce_set": "depmap_common_essential", **res})

    # Common-essential sensitivity
    for ce_name, ce_symbols_alt in common_essential_alternatives.items():
        ce_ids = symbols_to_geneids(ce_symbols_alt, sym2id)
        col = f"is_{ce_name}"
        df_std[col] = df_std["entrez_id"].isin(ce_ids).astype(int)
        res = run_full_model(df_std, "is_primary_liver_output", "is_primary_interface_context", ce_col=col)
        if res:
            model_rows.append({"lo_set": "primary_liver_output", "if_set": "primary_interface_context", "ce_set": ce_name, **res})

    s5 = pd.DataFrame(model_rows)
    s5.to_csv(SENS_DIR / "s5_model_sensitivity.csv", index=False)
    print(f"  Saved: {SENS_DIR / 's5_model_sensitivity.csv'}")
    plot_model_sensitivity_forest(
        s5,
        FIGS_DIR / "s5_model_sensitivity_forest_plot.pdf",
    )

    # ============================================================
    # STEP 4: Leave-one-term-out
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 4 — Leave-one-term-out sensitivity")
    print("=" * 60)

    if GO_BASIC_OBO.exists() and NCBI_GENE2GO.exists():
        from goatools.obo_parser import GODag
        from goatools.anno.genetogo_reader import Gene2GoReader

        obodag = GODag(str(GO_BASIC_OBO))
        g2g = Gene2GoReader(str(NCBI_GENE2GO), taxids=[TAXID_MOUSE])
        geneid2gos_bp = g2g.get_id2gos(namespace="BP")

        def build_term2geneids(annotated_universe: list[int], geneid2gos: dict[int, set[str]], obodag) -> dict[str, set[int]]:
            t2g: dict[str, set[int]] = {}
            for gid in annotated_universe:
                expanded: set[str] = set()
                for t in geneid2gos.get(gid, set()):
                    if t in obodag:
                        expanded.add(t)
                        expanded.update(obodag[t].get_all_parents())
                for t in expanded:
                    t2g.setdefault(t, set()).add(gid)
            return t2g

        annotated_univ = sorted(universe_set & set(geneid2gos_bp.keys()))
        term2gids = build_term2geneids(annotated_univ, geneid2gos_bp, obodag)

        loo_rows = []

        def safe_col(base: str, go_id: str) -> str:
            return f"{base}_{go_id.replace(':', '_')}"

        # Liver output
        for go_id in LIVER_OUTPUT_TERMS:
            reduced_ids: set[int] = set()
            for other_id in LIVER_OUTPUT_TERMS:
                if other_id == go_id:
                    continue
                reduced_ids |= term2gids.get(other_id, set())
            reduced_ids &= universe_set
            col = safe_col("is_lo_minus", go_id)
            df_std[col] = df_std["entrez_id"].isin(reduced_ids).astype(int)
            res = run_full_model(df_std, col, "is_primary_interface_context")
            if res:
                loo_rows.append({"left_out_term": go_id, "set": "liver_output", **res})

        # Interface
        for go_id in INTERFACE_PARENT_TERMS:
            reduced_ids = set()
            for other_id in INTERFACE_PARENT_TERMS:
                if other_id == go_id:
                    continue
                reduced_ids |= term2gids.get(other_id, set())
            reduced_ids &= universe_set
            col = safe_col("is_if_minus", go_id)
            df_std[col] = df_std["entrez_id"].isin(reduced_ids).astype(int)
            res = run_full_model(df_std, "is_primary_liver_output", col)
            if res:
                loo_rows.append({"left_out_term": go_id, "set": "interface_context", **res})

        s6 = pd.DataFrame(loo_rows)
        s6.to_csv(SENS_DIR / "s6_leave_one_term_out.csv", index=False)
        print(f"  Saved: {SENS_DIR / 's6_leave_one_term_out.csv'}")

    # ============================================================
    # STEP 5: Residual hit classification
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 5 — Residual hit classification")
    print("=" * 60)

    # Mechanistic annotation: GO enrichment in residual depleted set
    def residual_go_enrichment(residual_dep_ids: set[int], residual_univ_ids: list[int]) -> dict[int, str]:
        if not (GO_BASIC_OBO.exists() and NCBI_GENE2GO.exists()):
            return {}
        from goatools.go_enrichment import GOEnrichmentStudy
        obodag = GODag(str(GO_BASIC_OBO))
        g2g = Gene2GoReader(str(NCBI_GENE2GO), taxids=[TAXID_MOUSE])
        geneid2gos_bp = g2g.get_id2gos(namespace="BP")
        annotated = sorted(set(residual_univ_ids) & set(geneid2gos_bp.keys()))
        study = sorted(set(residual_dep_ids) & set(annotated))
        if not study:
            return {}
        goea = GOEnrichmentStudy(annotated, geneid2gos_bp, obodag, methods=["fdr_bh"], alpha=0.10)
        results = goea.run_study(study)
        sig = [r for r in results if r.p_fdr_bh is not None and r.p_fdr_bh < 0.10 and r.NS == "BP"]
        # Build gene -> top term map
        term2gids = build_term2geneids(annotated, geneid2gos_bp, obodag)
        gene_to_term: dict[int, str] = {}
        for r in sorted(sig, key=lambda x: x.p_fdr_bh):
            term_ids = term2gids.get(r.GO, set())
            for gid in term_ids & residual_dep_ids:
                if gid not in gene_to_term:
                    gene_to_term[gid] = f"{r.GO}:{r.name}"
        return gene_to_term

    residual_rows = []
    for ce_name, ce_symbols_alt in common_essential_alternatives.items():
        residual_dep = depleted - ce_symbols_alt
        residual_dep_ids = symbols_to_geneids(residual_dep, sym2id)
        residual_univ_ids = sorted((background_genes - ce_symbols_alt) & set(sym2id.keys()))
        mech = residual_go_enrichment(residual_dep_ids, [sym2id[g] for g in residual_univ_ids if g in sym2id])

        for gene in sorted(residual_dep):
            gid = sym2id.get(gene)
            if gid is None:
                continue
            is_lo = gene in primary_lo
            is_if = gene in primary_if
            if is_lo and is_if:
                category = "both"
            elif is_lo:
                category = "liver_output"
            elif is_if:
                category = "interface_context"
            else:
                category = "other"
            residual_rows.append({
                "gene": gene,
                "ce_definition": ce_name,
                "category": category,
                "interface_subcat": primary_if_subcat.get(gene, ""),
                "median_lfc": annot.loc[gene, EFFECT_COL] if gene in annot.index else np.nan,
                "fdr": annot.loc[gene, FDR_COL] if gene in annot.index else np.nan,
                "mechanistic_annotation": mech.get(gid, ""),
            })

    s7 = pd.DataFrame(residual_rows)
    s7.to_csv(SENS_DIR / "s7_residual_hit_classification.csv", index=False)
    print(f"  Saved: {SENS_DIR / 's7_residual_hit_classification.csv'}")

    # Summarise residual categories
    print("\n  Residual hit classification summary (primary CE = depmap):")
    summary = s7[s7["ce_definition"] == "depmap_common_essential"]["category"].value_counts()
    print(summary.to_string())


if __name__ == "__main__":
    main()
