#!/usr/bin/env python3
"""
analysis.py — Keys & Knouse mouse liver CRISPR screen
Decomposition: generic essentiality vs. interface/context vs. liver-output GO

Phases:
  1  Baseline GO enrichment
  2  Gene annotations: common-essential (DepMap), interface/context (GO),
     liver-output GO (narrow canonical hepatocyte output signature)
  3  Decomposition of depleted set (stacked bar)
  4  Rank-based distribution analysis (ECDF + violin) + domain visibility table
  5  Logistic regression model + forest plot
  6  Residual GO analysis (depleted minus common-essential)

External data (auto-downloaded if missing):
  - DepMap CRISPRInferredCommonEssentials.csv  (25Q3, post-Chronos)
  - GO / NCBI files (go-basic.obo, gene2go, gene_info)
  - MGI HOM_MouseHumanSequence.rpt  (genome-wide mouse–human homology)
"""

from __future__ import annotations

import gzip
import math
import shutil
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, mannwhitneyu, fisher_exact
import statsmodels.formula.api as smf
from goatools.go_enrichment import GOEnrichmentStudy
from goatools.obo_parser import GODag
from goatools.anno.genetogo_reader import Gene2GoReader


# ============================================================
# CONFIGURATION
# ============================================================

# --- Input files (local) ---
NEG_TABLE          = "mmc4_sup3_negative.csv"
POS_TABLE          = "mmc4_sup3_positive.csv"
SGRNA_COUNTS_TABLE = "SuppTable3_sgRNA_counts.csv"
HUMAN_MOUSE_TABLE  = "HOM_MouseHumanSequence.rpt"   # MGI vertebrate homology — full genome-wide map

GENE_COL   = "id"
FDR_COL    = "p.wilcox.bh"
EFFECT_COL = "median.lfc.all"

FDR_THRESHOLD = 0.05
EFFECT_EPS    = 0.0
TAXID_MOUSE   = 10090
GO_ALPHA      = 0.05
TOP_N_TERMS   = 12
MIN_GO_DEPTH  = 3

# --- Output directories ---
FIGS_DIR   = Path("figures")
TABLES_DIR = Path("tables")

# --- GO / NCBI files ---
GO_BASIC_OBO     = Path("go-basic.obo")
NCBI_GENE2GO     = Path("gene2go")
NCBI_GENE2GO_GZ  = Path("gene2go.gz")
NCBI_GENEINFO_GZ = Path("gene_info.gz")

# --- External annotation files ---
# CRISPRInferredCommonEssentials.csv: DepMap 25Q3 post-Chronos dependencies
# across all cell lines. From depmap.org if missing.
DEPMAP_CE_CSV = Path("CRISPRInferredCommonEssentials.csv")

# --- Download URLs ---
URL_GO_BASIC_OBO   = "http://purl.obolibrary.org/obo/go/go-basic.obo"
URL_GENE2GO_GZ     = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz"
URL_GENEINFO_GZ    = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz"

AUTO_DOWNLOAD = True

# ============================================================
# LIVER OUTPUT GO TERMS
# ============================================================
LIVER_OUTPUT_TERMS = [
    "GO:0006805",  # xenobiotic metabolic process
    "GO:0006699",  # bile acid biosynthetic process
    "GO:0006631",  # fatty acid metabolic process
    "GO:0008203",  # cholesterol metabolic process
    "GO:0007596",  # blood coagulation
    "GO:0006956",  # complement activation
    "GO:0009062",  # fatty acid catabolic process
    "GO:0006006",  # glucose metabolic process
]

# ============================================================
# INTERFACE / CONTEXT GO TERMS
# Parent terms are used — children are captured via GO ancestry expansion.
# Anchored to Keys & Knouse's explicitly identified in-vivo-unique hits:
#   MHC class I presentation + heparan sulfate biosynthesis.
# Extended to a candidate tissue-visible interface domain (ECM, cell adhesion).
# This defines the domain to be tested — not a prediction that it will be
# broadly depleted. Weak depletion across this domain would support sparse
# policing of the interface by endogenous in-vivo selection.
# Does NOT include generic signalling or transcription.
# ============================================================
INTERFACE_GO_TERMS = [
    # Antigen processing & presentation (Keys & Knouse MHC-I hits)
    "GO:0019882",  # antigen processing and presentation
    # Heparan sulfate / glycosaminoglycan (Keys & Knouse HS biosynthesis hits)
    "GO:0015012",  # heparan sulfate proteoglycan biosynthetic process
    "GO:0030203",  # glycosaminoglycan metabolic process
    # ECM organisation / extracellular structure
    "GO:0030198",  # extracellular matrix organization
    "GO:0043062",  # extracellular structure organization
    # Cell adhesion / tissue embedding
    "GO:0007155",  # cell adhesion
    "GO:0098609",  # cell-cell adhesion
    "GO:0007160",  # cell-matrix adhesion
]

# Subcategory sets (subset of INTERFACE_GO_TERMS)
INTERFACE_IMMUNE   = {"GO:0019882"}
INTERFACE_ECM_HS   = {"GO:0015012", "GO:0030203", "GO:0030198", "GO:0043062"}
INTERFACE_ADHESION = {"GO:0007155", "GO:0098609", "GO:0007160"}

# Decomposition colour palette
CAT_COLORS = {
    "common_essential": "#e15759",
    "interface":        "#4e79a7",
    "liver_specific":   "#76b7b2",
    "other":            "#bab0ac",
}
CAT_LABELS = {
    "common_essential": "Common essential",
    "interface":        "Interface/context",
    "liver_specific":   "Liver-output GO",
    "other":            "Other",
}

# ============================================================
# UTILITIES
# ============================================================

def ensure_dirs() -> None:
    FIGS_DIR.mkdir(exist_ok=True)
    TABLES_DIR.mkdir(exist_ok=True)


def download_if_missing(path: Path, url: str, label: str = "") -> bool:
    if path.exists():
        return True
    if not AUTO_DOWNLOAD:
        return False
    lbl = label or str(path)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=90) as resp, open(path, "wb") as out:
            shutil.copyfileobj(resp, out)
        return True
    except Exception as exc:
        return False


def gunzip_to_plain(gz_path: Path, out_path: Path) -> None:
    if out_path.exists():
        return
    with gzip.open(gz_path, "rb") as fin, open(out_path, "wb") as fout:
        shutil.copyfileobj(fin, fout)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in (GENE_COL, FDR_COL, EFFECT_COL):
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {path}")
    return df


def extract_technical_covariates(counts_csv: str) -> pd.DataFrame:
    df = pd.read_csv(counts_csv)
    cov = df.groupby("gene").agg(
        sgrna_count=("sgRNA", "count"),
        baseline_abundance=("plasmid.ttgact", "median"),
    ).reset_index()
    cov.rename(columns={"gene": GENE_COL}, inplace=True)
    cov["log2_abundance"] = np.log2(cov["baseline_abundance"] + 1)
    return cov


def build_human_to_mouse(path: str) -> dict[str, str]:
    """
    Parse MGI HOM_MouseHumanSequence.rpt (tab-delimited, genome-wide homology).
    Columns: DB Class Key, Common Organism Name, NCBI Taxon ID, Symbol, etc
    Groups entries by DB Class Key; maps human symbol to mouse symbol (1:1 pairs).
    One-to-many homologs are included (human symbol maps to first/only mouse symbol).
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    key_col    = "DB Class Key"
    taxon_col  = "NCBI Taxon ID"
    symbol_col = "Symbol"
    mouse = (
        df[df[taxon_col] == "10090"][[key_col, symbol_col]]
        .rename(columns={symbol_col: "mouse_symbol"})
        .dropna())
    human = (
        df[df[taxon_col] == "9606"][[key_col, symbol_col]]
        .rename(columns={symbol_col: "human_symbol"})
        .dropna())
    merged = human.merge(mouse, on=key_col, how="inner")
    merged = merged.drop_duplicates(subset=["human_symbol"])
    result = dict(zip(merged["human_symbol"].str.strip(), merged["mouse_symbol"].str.strip()))
    return result


def build_symbol2geneid(gene_info_gz: Path, taxid: int) -> dict[str, int]:
    official: dict[str, int] = {}
    syn_to_gids: dict[str, set[int]] = defaultdict(set)
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


# ── GO helpers ──────────────────────────────────────────────

def go_results_to_df(go_results) -> pd.DataFrame:
    rows = []
    for r in go_results:
        gr = (r.study_count / r.study_n) if r.study_n else 0.0
        rows.append({
            "GO": r.GO,
            "name": r.name,
            "p_fdr_bh": r.p_fdr_bh,
            "study_count": r.study_count,
            "study_n": r.study_n,
            "gene_ratio": gr,
        })
    df = pd.DataFrame(rows)
    if len(df):
        nz = df.loc[df["p_fdr_bh"] > 0, "p_fdr_bh"].min()
        if pd.isna(nz):
            nz = 1e-300
        df["p_clamped"] = df["p_fdr_bh"].apply(lambda p: p if p > 0 else nz / 10.0)
        df["neglog10_fdr"] = df["p_clamped"].apply(lambda p: -math.log10(p))
        df = df.sort_values(["p_fdr_bh", "study_count"], ascending=[True, False])
    return df


def filter_by_min_depth(go_df: pd.DataFrame, obodag, min_depth: int) -> pd.DataFrame:
    if go_df is None or go_df.empty:
        return go_df
    depths = [obodag[g].depth if g in obodag else 0 for g in go_df["GO"]]
    out = go_df.copy()
    out["depth"] = depths
    return out[out["depth"] >= min_depth].sort_values(["p_fdr_bh", "study_count"], ascending=[True, False])


def run_go_bp(
    study_ids: set[int],
    annotated_universe: list[int],
    universe_set: set[int],
    geneid2gos_bp,
    obodag,
):
    study = sorted(study_ids & universe_set)
    if not study:
        return []
    goea = GOEnrichmentStudy(
        annotated_universe, geneid2gos_bp, obodag,
        methods=["fdr_bh"], alpha=GO_ALPHA,
    )
    results = goea.run_study(study)
    return [
        r for r in results
        if r.p_fdr_bh is not None and r.p_fdr_bh < GO_ALPHA and r.NS == "BP"
    ]


def build_term2geneids(
    annotated_universe: list[int],
    geneid2gos_bp: dict[int, set[str]],
    obodag,
) -> dict[str, set[int]]:
    t2g: dict[str, set[int]] = {}
    for gid in annotated_universe:
        expanded: set[str] = set()
        for t in geneid2gos_bp.get(gid, set()):
            if t in obodag:
                expanded.add(t)
                expanded.update(obodag[t].get_all_parents())
        for t in expanded:
            t2g.setdefault(t, set()).add(gid)
    return t2g


def fisher_underrep(
    study: set[int], term_genes: set[int], universe: set[int]
) -> tuple[int, int, int, int, float, float]:
    a = len(study & term_genes)
    b = len(study - term_genes)
    c = len((universe - study) & term_genes)
    d = len((universe - study) - term_genes)
    odds, p = fisher_exact([[a, b], [c, d]], alternative="less")
    return a, b, c, d, odds, p


def fisher_overrep(
    study: set[int], term_genes: set[int], universe: set[int]
) -> tuple[int, int, int, int, float, float]:
    a = len(study & term_genes)
    b = len(study - term_genes)
    c = len((universe - study) & term_genes)
    d = len((universe - study) - term_genes)
    odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
    return a, b, c, d, odds, p


def abbreviate_term(term: str) -> str:
    replacements = {
        "RNA splicing, via transesterification": "RNA splicing (transester.)",
        "antigen processing and presentation": "antigen proc./present.",
        "negative regulation": "neg. reg.",
        "positive regulation": "pos. reg.",
        "regulation": "reg.",
        "process": "proc.",
        "metabolic": "met.",
        "biosynthetic": "biosyn.",
        "catabolic": "cat.",
        "response": "resp.",
        "organization": "org.",
        "signaling": "sig.",
        "development": "dev.",
        "localization": "loc.",
        "transport": "trans.",
        "cellular": "cell.",
        "macromolecule": "macromol.",
        "biogenesis": "biogen.",
    }
    t = term
    for old, new in replacements.items():
        t = t.replace(old, new)
    return t


# ============================================================
# FIGURE HELPERS
# ============================================================

def save_go_dotplot(
    df: pd.DataFrame,
    out_file: Path,
    color: str,
    title: str = "",
) -> None:
    if df is None or df.empty:
        print(f"  [skip] {out_file.name} (no significant terms)")
        return
    d = df.head(TOP_N_TERMS).iloc[::-1].copy()
    ypos   = np.arange(len(d))
    xvals  = d["neglog10_fdr"].to_numpy()
    sizes  = d["gene_ratio"].to_numpy() * 3000
    labels = [abbreviate_term(t) for t in d["name"]]
    fig_h  = max(2.0, 0.35 * len(d) + 0.8)
    fig, ax = plt.subplots(figsize=(5.5, fig_h), dpi=200)
    ax.scatter(xvals, ypos, s=sizes, c=color, alpha=0.75, edgecolors="none")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel(r"$-\log_{10}(\mathrm{FDR})$", fontsize=10)
    if title:
        ax.set_title(title, fontsize=10, pad=6)
    if len(xvals):
        ax.set_xlim(0, xvals.max() * 1.15)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_file}")


# ============================================================
# PHASE 2 HELPERS: LOAD EXTERNAL ANNOTATIONS
# ============================================================

def load_depmap_common_essentials(
    path: Path, human_to_mouse: dict[str, str]
) -> set[str]:
    """Return mouse gene symbols identified as DepMap common-essential."""
    if not path.exists():
        print(f"  [!] {path} not found — common-essential annotation unavailable.")
        return set()
    df = pd.read_csv(path, header=None, names=["entry"])
    symbols: set[str] = set()
    for entry in df["entry"].dropna().astype(str):
        sym = entry.split("(")[0].strip()
        if sym:
            symbols.add(sym)
    mouse_symbols = {human_to_mouse[s] for s in symbols if s in human_to_mouse}
    print(
        f"  DepMap CE: {len(symbols)} human symbols → {len(mouse_symbols)} mouse orthologs"
    )
    return mouse_symbols


def build_interface_geneset(
    annotated_universe: list[int],
    geneid2gos_bp: dict[int, set[str]],
    obodag,
) -> tuple[set[int], dict[int, str]]:
    """
    Expand each interface GO term to all its descendants, then collect
    every gene in the annotated universe that belongs to ≥1 interface term.
    Returns (interface_geneids, gid → subcategory_string).
    NOTE: Returns gene IDs, not symbols. Caller maps to screen symbols.
    """
    immune_ids: set[str] = set()
    ecm_hs_ids: set[str] = set()
    adhesion_ids: set[str] = set()
    all_ids: set[str] = set()

    for term_id in INTERFACE_GO_TERMS:
        if term_id not in obodag:
            print(f"  [!] Interface GO term {term_id} not found in obodag")
            continue
        term = obodag[term_id]
        descendants = {term_id} | term.get_all_children()
        all_ids |= descendants
        if term_id in INTERFACE_IMMUNE:
            immune_ids |= descendants
        if term_id in INTERFACE_ECM_HS:
            ecm_hs_ids |= descendants
        if term_id in INTERFACE_ADHESION:
            adhesion_ids |= descendants

    interface_geneids: set[int] = set()
    gid_subcategory: dict[int, str] = {}

    for gid in annotated_universe:
        expanded: set[str] = set()
        for t in geneid2gos_bp.get(gid, set()):
            if t in obodag:
                expanded.add(t)
                expanded.update(obodag[t].get_all_parents())
        if expanded & all_ids:
            interface_geneids.add(gid)
            if expanded & immune_ids:
                gid_subcategory[gid] = "immune_interface"
            elif expanded & ecm_hs_ids:
                gid_subcategory[gid] = "ecm_hs_interface"
            else:
                gid_subcategory[gid] = "adhesion_interface"

    counts = {
        "immune":   sum(1 for v in gid_subcategory.values() if v == "immune_interface"),
        "ecm_hs":   sum(1 for v in gid_subcategory.values() if v == "ecm_hs_interface"),
        "adhesion": sum(1 for v in gid_subcategory.values() if v == "adhesion_interface"),
    }
    print(
        f"  Interface gene IDs: {len(interface_geneids)} total  "
        f"(immune={counts['immune']}, ECM/HS={counts['ecm_hs']}, adhesion={counts['adhesion']})"
    )
    return interface_geneids, gid_subcategory


# ============================================================
# PHASE 3: DECOMPOSITION
# ============================================================

def decompose_gene_set(
    gene_set: set[str],
    is_common_essential: set[str],
    is_interface: set[str],
    is_liver_specific: set[str],
) -> dict[str, set[str]]:
    """
    Partition genes into mutually exclusive categories (priority order):
      common_essential > interface > liver_specific > other
    """
    ce        = gene_set & is_common_essential
    residual1 = gene_set - ce
    iface     = residual1 & is_interface
    residual2 = residual1 - iface
    liver     = residual2 & is_liver_specific
    other     = residual2 - liver
    return {
        "common_essential": ce,
        "interface":        iface,
        "liver_specific":   liver,
        "other":            other,
    }


def plot_decomposition_bar(
    decomp: dict[str, set[str]],
    total: int,
    out_file: Path,
    title: str = "",
) -> None:
    cats   = ["common_essential", "interface", "liver_specific", "other"]
    counts = [len(decomp[c]) for c in cats]

    fig, ax = plt.subplots(figsize=(6, 2.2), dpi=200)
    left = 0.0
    for cat, cnt in zip(cats, counts):
        pct = cnt / total * 100 if total else 0.0
        ax.barh(
            0, pct, left=left, color=CAT_COLORS[cat],
            label=f"{CAT_LABELS[cat]} ({cnt})"
        )
        if pct > 3.5:
            ax.text(
                left + pct / 2, 0,
                f"{pct:.0f}%",
                ha="center", va="center", fontsize=9,
                color="white", fontweight="bold",
            )
        left += pct

    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("% of depleted genes", fontsize=10)
    if title:
        ax.set_title(title, fontsize=10, pad=6)
    ax.legend(
        loc="upper center", 
        bbox_to_anchor=(0.5, -0.40),
        ncol=2, 
        fontsize=8, 
        frameon=False,
        columnspacing=2.0,
        labelspacing=1.2 
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_file}")


# ============================================================
# PHASE 4: RANKED DISTRIBUTIONS
# ============================================================

def plot_ranked_distributions(
    annot_df: pd.DataFrame,
    lfc_col: str,
    out_file: Path,
) -> None:
    """
    Left panel: overlaid ECDFs of LFC per category.
    Right panel: violin plots of LFC per category.
    annot_df must have category and lfc_col columns.
    """
    cat_order  = ["common_essential", "interface", "liver_specific", "background"]
    cat_labels = [CAT_LABELS.get(c, c) for c in cat_order]
    cat_colors = [CAT_COLORS.get(c, "#aaa") for c in cat_order]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=200)

    # --- ECDF ---
    ax = axes[0]
    for cat, col, lbl in zip(cat_order, cat_colors, cat_labels):
        vals = annot_df.loc[annot_df["category"] == cat, lfc_col].dropna().values
        if len(vals) < 2:
            continue
        sv   = np.sort(vals)
        ecdf = np.arange(1, len(sv) + 1) / len(sv)
        ax.plot(sv, ecdf, color=col, linewidth=1.8,
                label=f"{lbl} (n={len(vals)})")
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Median LFC  (depleted ←)", fontsize=10)
    ax.set_ylabel("Cumulative fraction", fontsize=10)
    ax.set_title("Depletion score ECDF by gene category", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Violin ---
    ax = axes[1]
    data_valid   = []
    labels_valid = []
    colors_valid = []
    for cat, col, lbl in zip(cat_order, cat_colors, cat_labels):
        vals = annot_df.loc[annot_df["category"] == cat, lfc_col].dropna().values
        if len(vals) >= 5:
            data_valid.append(vals)
            labels_valid.append(lbl)
            colors_valid.append(col)

    if data_valid:
        positions = range(len(data_valid))
        parts = ax.violinplot(
            data_valid, positions=positions,
            showmedians=True, showextrema=False
        )
        for pc, col in zip(parts["bodies"], colors_valid):
            pc.set_facecolor(col)
            pc.set_alpha(0.70)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)
        ax.set_xticks(list(positions))
        ax.set_xticklabels(labels_valid, fontsize=9)
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_ylabel("Median LFC", fontsize=10)
        ax.set_title("Distribution of depletion score", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_file}")


# ============================================================
# PHASE 5: LOGISTIC REGRESSION + FOREST PLOT
# ============================================================

def run_logistic_regression(df_model: pd.DataFrame, out_file: Path) -> None:
    """
    Fit nested logistic regression models predicting is_depleted.
    Required columns: is_depleted, sgrna_count, log2_abundance,
                      is_common_essential, is_interface, is_liver_output_go
    """
    required = [
        "is_depleted",
        "sgrna_count",
        "log2_abundance",
        "is_common_essential",
        "is_interface",
        "is_liver_output_go",
    ]
    missing = [c for c in required if c not in df_model.columns]
    if missing:
        print(f"  [!] Logistic regression skipped — missing columns: {missing}")
        return

    df = df_model[required].dropna()
    if df["is_depleted"].sum() < 10:
        print("  [!] Too few depleted genes for regression (<10).")
        return

    for col in ["sgrna_count", "log2_abundance"]:
        sd = df[col].std()
        if sd > 0:
            df = df.copy()
            df[col] = (df[col] - df[col].mean()) / sd

    nested_specs = [
        ("Technical only",
         "is_depleted ~ sgrna_count + log2_abundance"),
        ("+ Common essential",
         "is_depleted ~ sgrna_count + log2_abundance + is_common_essential"),
        ("+ Liver-output GO",
         "is_depleted ~ sgrna_count + log2_abundance + is_common_essential + is_liver_output_go"),
        ("Full model",
         "is_depleted ~ sgrna_count + log2_abundance + is_common_essential + is_liver_output_go + is_interface"),
    ]

    focus_terms = ["is_common_essential", "is_liver_output_go", "is_interface"]
    coef_rows   = []
    full_model  = None

    print("\n" + "=" * 60)
    print("PHASE 5 — Logistic regression results")
    print("=" * 60)

    for name, formula in nested_specs:
        try:
            m = smf.logit(formula=formula, data=df).fit(disp=0)
            print(f"\n  [{name}]  AIC={m.aic:.1f}  BIC={m.bic:.1f}  n={int(m.nobs)}")
            for term in focus_terms:
                if term in m.params:
                    b   = m.params[term]
                    se  = m.bse[term]
                    p   = m.pvalues[term]
                    OR  = np.exp(b)
                    lo  = np.exp(b - 1.96 * se)
                    hi  = np.exp(b + 1.96 * se)
                    print(f"    {term:<28}  OR={OR:.2f} [{lo:.2f}–{hi:.2f}]  p={p:.2e}")
                    coef_rows.append({
                        "model": name,
                        "term": term,
                        "coef": b,
                        "se": se,
                        "p": p,
                        "OR": OR,
                        "CI_lo": lo,
                        "CI_hi": hi,
                    })
            if name == "Full model":
                full_model = m
        except Exception as exc:
            print(f"  [!] Model '{name}' failed: {exc}")

    if coef_rows:
        pd.DataFrame(coef_rows).to_csv(
            TABLES_DIR / "p5_logistic_coefficients.csv", index=False
        )
        print(f"\n  Saved: {TABLES_DIR / 'p5_logistic_coefficients.csv'}")

    if full_model is not None:
        _plot_forest(full_model, focus_terms, out_file)


def _plot_forest(model, highlight_terms: list[str], out_file: Path) -> None:
    label_map = {
        "is_common_essential": "Common essential",
        "is_liver_output_go":  "Liver-output GO",
        "is_interface":        "Interface/context",
        "sgrna_count":         "sgRNA count (z-score)",
        "log2_abundance":      "Log₂ baseline abundance (z-score)",
    }
    rows = []
    for term in model.params.index:
        if term == "Intercept":
            continue
        b  = model.params[term]
        se = model.bse[term]
        p  = model.pvalues[term]
        rows.append({
            "label":     label_map.get(term, term),
            "OR":        np.exp(b),
            "CI_lo":     np.exp(b - 1.96 * se),
            "CI_hi":     np.exp(b + 1.96 * se),
            "p":         p,
            "highlight": term in highlight_terms,
        })

    df = pd.DataFrame(rows).iloc[::-1]

    fig, ax = plt.subplots(figsize=(5.5, max(2.5, 0.55 * len(df) + 1.0)), dpi=200)
    for i, (_, row) in enumerate(df.iterrows()):
        col = "#e15759" if row["OR"] > 1 else "#4e79a7"
        lw  = 2.2 if row["highlight"] else 1.2
        ms  = 80 if row["highlight"] else 50
        ax.plot(
            [row["CI_lo"], row["CI_hi"]], [i, i],
            color=col, linewidth=lw, solid_capstyle="round", zorder=3)
        ax.scatter([row["OR"]], [i], color=col, s=ms, zorder=5)
        p_str = f"p={row['p']:.1e}" if row["p"] < 0.01 else f"p={row['p']:.3f}"
        ax.text(row["CI_hi"] * 1.3, i, p_str, va="center", fontsize=8)

    ax.set_xscale("log")
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.9, zorder=1)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["label"].tolist(), fontsize=9)
    ax.set_xlabel("Odds ratio (95 % CI)", fontsize=10)
    ax.set_title("Predictors of depletion — full logistic model", fontsize=10, pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_file}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ensure_dirs()

    # ── Load screen data ─────────────────────────────────────
    print("=" * 60)
    print("Loading screen data ...")
    print("=" * 60)

    neg = load_csv(NEG_TABLE)
    pos = load_csv(POS_TABLE)
    neg[GENE_COL] = neg[GENE_COL].astype(str)
    pos[GENE_COL] = pos[GENE_COL].astype(str)

    if Path(SGRNA_COUNTS_TABLE).exists():
        cov_df = extract_technical_covariates(SGRNA_COUNTS_TABLE)
        neg = neg.merge(cov_df, on=GENE_COL, how="left")

    background_genes: set[str] = set(neg[GENE_COL].unique())
    sig_neg = neg.loc[neg[FDR_COL] < FDR_THRESHOLD]
    sig_pos = pos.loc[pos[FDR_COL] < FDR_THRESHOLD]

    depleted: set[str] = set(
        sig_neg.loc[sig_neg[EFFECT_COL] < -EFFECT_EPS, GENE_COL].unique()
    )
    enriched: set[str] = set(
        sig_pos.loc[sig_pos[EFFECT_COL] > EFFECT_EPS, GENE_COL].unique()
    )

    print(f"  Universe: {len(background_genes)}  |  Depleted: {len(depleted)}  |  Enriched: {len(enriched)}")

    # ── GO setup ─────────────────────────────────────────────
    download_if_missing(GO_BASIC_OBO, URL_GO_BASIC_OBO)
    if not NCBI_GENE2GO.exists():
        download_if_missing(NCBI_GENE2GO_GZ, URL_GENE2GO_GZ)
        gunzip_to_plain(NCBI_GENE2GO_GZ, NCBI_GENE2GO)
    download_if_missing(NCBI_GENEINFO_GZ, URL_GENEINFO_GZ)

    # # # #

    obodag        = GODag(str(GO_BASIC_OBO))
    g2g           = Gene2GoReader(str(NCBI_GENE2GO), taxids=[TAXID_MOUSE])
    geneid2gos_bp = g2g.get_id2gos(namespace="BP")
    sym2id        = build_symbol2geneid(NCBI_GENEINFO_GZ, TAXID_MOUSE)

    univ_ids = symbols_to_geneids(background_genes, sym2id)
    dep_ids  = symbols_to_geneids(depleted, sym2id)
    enr_ids  = symbols_to_geneids(enriched, sym2id)

    annotated_univ = sorted(univ_ids & set(geneid2gos_bp.keys()))
    univ_set       = set(annotated_univ)

    # Mapping from screen gene symbol → gene ID (used for GO-derived annotations)
    # This avoids the id→sym reverse mapping bug (synonyms vs official symbols)
    screen_sym_to_id: dict[str, int] = {
        g: sym2id[g] for g in background_genes if g in sym2id
    }

    id_to_screen_sym: dict[int, str] = {
        gid: sym for sym, gid in screen_sym_to_id.items()
    }

    # ── PHASE 1: Baseline GO ─────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 1 — Baseline GO enrichment")
    print("=" * 60)

    p1_dep_cache = TABLES_DIR / "p1_go_depleted_baseline.csv"
    p1_enr_cache = TABLES_DIR / "p1_go_enriched_baseline.csv"

    if p1_dep_cache.exists() and p1_enr_cache.exists():
        print("  [cache] Loading Phase 1 GO results from tables/")
        go_dep_df = pd.read_csv(p1_dep_cache)
        go_enr_df = pd.read_csv(p1_enr_cache)
    else:
        go_dep = run_go_bp(dep_ids, annotated_univ, univ_set, geneid2gos_bp, obodag)
        go_enr = run_go_bp(enr_ids, annotated_univ, univ_set, geneid2gos_bp, obodag)
        go_dep_df = filter_by_min_depth(go_results_to_df(go_dep), obodag, MIN_GO_DEPTH)
        go_enr_df = filter_by_min_depth(go_results_to_df(go_enr), obodag, MIN_GO_DEPTH)
        go_dep_df.to_csv(p1_dep_cache, index=False)
        go_enr_df.to_csv(p1_enr_cache, index=False)

    save_go_dotplot(
        go_dep_df,
        FIGS_DIR / "p1_go_depleted_baseline.pdf",
        "#d62728",
        "Depleted GO (baseline)",
    )
    save_go_dotplot(
        go_enr_df,
        FIGS_DIR / "p1_go_enriched_baseline.pdf",
        "#2ca02c",
        "Enriched GO (baseline)",
    )

    # Liver-output under-representation (baseline)
    term2gids = build_term2geneids(annotated_univ, geneid2gos_bp, obodag)
    liver_union_ids: set[int] = set()
    for go_id in LIVER_OUTPUT_TERMS:
        liver_union_ids |= term2gids.get(go_id, set())

    dep_in_univ = dep_ids & univ_set
    a0, _, _, _, odds0, p0 = fisher_underrep(dep_in_univ, liver_union_ids, univ_set)
    print(
        f"  Liver-output under-rep (pooled):  "
        f"OR={odds0:.3f}, p={p0:.2e}  "
        f"({a0}/{len(dep_in_univ)} depleted genes are liver-output)"
    )

    # ── PHASE 2: External annotations ────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2 — Building gene annotations")
    print("=" * 60)

    # Use a versioned cache so old HPA-derived caches do not leak through
    p2_cache = TABLES_DIR / "p2_gene_annotations_v2.csv"

    if p2_cache.exists():
        print(f"  [cache] Loading Phase 2 annotations from {p2_cache}")
        annot = pd.read_csv(p2_cache)

        ce_symbols               = set(annot.loc[annot["is_common_essential"] == 1, "gene"])
        liver_output_go_symbols  = set(annot.loc[annot["is_liver_output_go"] == 1, "gene"])
        interface_screen_symbols = set(annot.loc[annot["is_interface"] == 1, "gene"])

        interface_subcats_sym: dict[str, str] = (
            annot.dropna(subset=["interface_subcat"])
            .set_index("gene")["interface_subcat"]
            .to_dict()
        )

        liver_specific_symbols = liver_output_go_symbols
    else:
        human_to_mouse = build_human_to_mouse(HUMAN_MOUSE_TABLE)
        print(f"  Human→mouse ortholog map: {len(human_to_mouse)} entries")

        # DepMap common essentials
        if not DEPMAP_CE_CSV.exists():
            print(f"  [!] {DEPMAP_CE_CSV} not found. Download from depmap.org.")
        ce_symbols = load_depmap_common_essentials(DEPMAP_CE_CSV, human_to_mouse)

        # Interface genes from GO — returns gene IDs (avoids id→sym synonym mismatch)
        p2_iface_cache = TABLES_DIR / "p2_interface_geneids.csv"
        if p2_iface_cache.exists():
            print("  [cache] Loading interface gene IDs from tables/")
            iface_df = pd.read_csv(p2_iface_cache)
            interface_geneids = set(iface_df["geneid"])
            gid_subcategory   = iface_df.set_index("geneid")["subcat"].to_dict()
        else:
            interface_geneids, gid_subcategory = build_interface_geneset(
                annotated_univ, geneid2gos_bp, obodag
            )
            pd.DataFrame([
                {"geneid": gid, "subcat": sub}
                for gid, sub in gid_subcategory.items()
            ]).to_csv(p2_iface_cache, index=False)

        interface_screen_symbols: set[str] = {
            g for g, gid in screen_sym_to_id.items() if gid in interface_geneids
        }
        interface_subcats_sym: dict[str, str] = {
            g: gid_subcategory[gid]
            for g, gid in screen_sym_to_id.items()
            if gid in gid_subcategory
        }
        print(
            f"  Interface screen genes: {len(interface_screen_symbols)} "
            f"(matched from {len(interface_geneids)} GO-annotated gene IDs)"
        )

        # Liver-output GO symbols — map via screen's sym→id
        liver_output_go_symbols: set[str] = {
            g for g, gid in screen_sym_to_id.items() if gid in liver_union_ids
        }
        liver_specific_symbols = liver_output_go_symbols

        annot = (
            neg[[GENE_COL, EFFECT_COL, FDR_COL]]
            .rename(columns={GENE_COL: "gene"})
            .copy()
        )
        for col in ["sgrna_count", "log2_abundance"]:
            if col in neg.columns:
                annot = annot.merge(
                    neg[[GENE_COL, col]].rename(columns={GENE_COL: "gene"}),
                    on="gene",
                    how="left",
                )

        annot["is_depleted"]         = annot["gene"].isin(depleted).astype(int)
        annot["is_enriched"]         = annot["gene"].isin(enriched).astype(int)
        annot["is_common_essential"] = annot["gene"].isin(ce_symbols).astype(int)
        annot["is_liver_output_go"]  = annot["gene"].isin(liver_output_go_symbols).astype(int)
        annot["is_interface"]        = annot["gene"].isin(interface_screen_symbols).astype(int)
        annot["interface_subcat"]    = annot["gene"].map(interface_subcats_sym)

        annot.to_csv(p2_cache, index=False)

    print(f"\n  Annotation summary (n={len(annot)} universe genes):")
    for col in [
        "is_depleted",
        "is_enriched",
        "is_common_essential",
        "is_liver_output_go",
        "is_interface",
    ]:
        n   = int(annot[col].sum())
        pct = n / len(annot) * 100
        print(f"    {col:<28}: {n:>5} ({pct:5.1f}%)")

    # ── PHASE 3: Decomposition ────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3 — Decomposition of depleted set")
    print("=" * 60)

    decomp = decompose_gene_set(
        depleted, ce_symbols, interface_screen_symbols, liver_specific_symbols
    )
    print(f"  Depleted total: {len(depleted)}")
    for cat, genes in decomp.items():
        pct = len(genes) / len(depleted) * 100 if depleted else 0.0
        print(f"    {cat:<28}: {len(genes):>4}  ({pct:.1f}%)")
        if cat == "interface" and genes:
            # Attach the subcategory annotation to each gene
            annotated = [f"{g} ({interface_subcats_sym.get(g, 'unknown')})" for g in sorted(genes)]
            print(f"      -> Genes: {', '.join(annotated)}")
        elif cat == "liver_specific" and genes:
            print(f"      -> Genes: {', '.join(sorted(genes))}")

    plot_decomposition_bar(
        decomp,
        len(depleted),
        FIGS_DIR / "p3_depleted_decomposition.pdf",
        f"Composition of depleted gene set (n={len(depleted)})",
    )

    decomp_rows = [
        {"gene": g, "category": cat}
        for cat, genes in decomp.items()
        for g in sorted(genes)
    ]
    pd.DataFrame(decomp_rows).to_csv(
        TABLES_DIR / "p3_depleted_decomposition.csv", index=False
    )

    # ── PHASE 4: Ranked distributions ────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 4 — Ranked distribution analysis")
    print("=" * 60)

    def assign_category(g: str) -> str:
        if g in ce_symbols:
            return "common_essential"
        elif g in interface_screen_symbols:
            return "interface"
        elif g in liver_specific_symbols:
            return "liver_specific"
        else:
            return "background"

    annot["category"] = annot["gene"].apply(assign_category)

    cat_summary = (
        annot.groupby("category")["is_depleted"]
        .agg(n_depleted="sum", n_total="count")
        .reset_index()
    )
    cat_summary["depletion_rate"] = cat_summary["n_depleted"] / cat_summary["n_total"]
    print(cat_summary.to_string(index=False))
    cat_summary.to_csv(TABLES_DIR / "p4_domain_visibility.csv", index=False)

    bg_lfc = annot.loc[annot["category"] == "background", EFFECT_COL].dropna().values
    ks_rows = []
    print()
    for cat in ["common_essential", "interface", "liver_specific"]:
        cat_lfc = annot.loc[annot["category"] == cat, EFFECT_COL].dropna().values
        if len(cat_lfc) < 5:
            continue
        # scipy KS: alternative='greater' → stat=max(F1-F2) → tests if x1 CDF > x2 CDF
        # i.e. x1 stochastically LESS than x2 (more negative / more depleted)
        # mannwhitneyu: alternative='less' → tests if x1 values < x2 values (same direction)
        ks_stat, ks_p = ks_2samp(cat_lfc, bg_lfc, alternative="greater")
        mw_stat, mw_p = mannwhitneyu(cat_lfc, bg_lfc, alternative="less")
        median_diff = np.median(cat_lfc) - np.median(bg_lfc)
        print(
            f"  {cat:<28}  n={len(cat_lfc):<5}  "
            f"KS p={ks_p:.2e}  MW p={mw_p:.2e}  "
            f"median LFC shift={median_diff:+.3f}"
        )
        ks_rows.append({
            "category": cat,
            "n": len(cat_lfc),
            "ks_p": ks_p,
            "mw_p": mw_p,
            "median_lfc_shift": median_diff,
        })
    pd.DataFrame(ks_rows).to_csv(TABLES_DIR / "p4_distribution_tests.csv", index=False)

    plot_ranked_distributions(
        annot[["gene", EFFECT_COL, "category"]].rename(columns={EFFECT_COL: "lfc"}),
        "lfc",
        FIGS_DIR / "p4_ranked_distributions.pdf",
    )

    # ── PHASE 5: Logistic regression ─────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 5 — Logistic regression")
    print("=" * 60)

    run_logistic_regression(
        annot.rename(columns={EFFECT_COL: "lfc"}),
        FIGS_DIR / "p5_forest_plot.pdf",
    )

    # ── PHASE 6: Residual GO ──────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 6 — Residual GO (depleted minus common-essential)")
    print("=" * 60)

    residual_depleted = depleted - ce_symbols
    residual_dep_ids  = symbols_to_geneids(residual_depleted, sym2id)

    # Residual universe: exclude common-essential genes from background too
    ce_ids            = symbols_to_geneids(ce_symbols, sym2id)
    residual_univ_ids = univ_ids - ce_ids
    residual_annotated = sorted(residual_univ_ids & set(geneid2gos_bp.keys()))
    residual_univ_set  = set(residual_annotated)

    p6_cache = TABLES_DIR / "p6_go_residual_depleted.csv"

    if p6_cache.exists():
        print("  [cache] Loading Phase 6 GO results from tables/")
        go_res_df = pd.read_csv(p6_cache)
    else:
        go_residual = run_go_bp(
            residual_dep_ids,
            residual_annotated,
            residual_univ_set,
            geneid2gos_bp,
            obodag,
        )
        go_res_df = filter_by_min_depth(
            go_results_to_df(go_residual), obodag, MIN_GO_DEPTH
        )
        go_res_df.to_csv(p6_cache, index=False)

    print(
        f"  Residual depleted: {len(residual_depleted)} genes  "
        f"→ {len(go_res_df)} enriched GO terms"
    )

    save_go_dotplot(
        go_res_df,
        FIGS_DIR / "p6_go_residual_depleted.pdf",
        "#9467bd",
        f"Residual depleted GO (CE removed, n={len(residual_depleted)})",
    )

    # Test: is liver-output still under-represented in residual?
    res_dep_set = residual_dep_ids & residual_univ_set
    res_term2gids = build_term2geneids(residual_annotated, geneid2gos_bp, obodag)

    liver_union_res: set[int] = set()
    for go_id in LIVER_OUTPUT_TERMS:
        liver_union_res |= res_term2gids.get(go_id, set())

    a_liver, _, _, _, odds_liver, p_liver = fisher_underrep(
        res_dep_set, liver_union_res, residual_univ_set
    )
    print(
        f"  Liver-output under-rep in residual:  "
        f"OR={odds_liver:.3f}, p={p_liver:.2e}  "
        f"({a_liver}/{len(res_dep_set)} residual depleted are liver-output)"
    )
    print("Phase 6 liver depleted genes:")
    p6_liver_ids = res_dep_set & liver_union_res
    p6_liver_syms = sorted([id_to_screen_sym[gid] for gid in p6_liver_ids if gid in id_to_screen_sym])
    print(f"    -> Genes: {', '.join(p6_liver_syms)}")

    print("Phase 1 liver depleted genes:")
    p1_liver_ids = dep_in_univ & liver_union_ids
    p1_liver_syms = sorted([id_to_screen_sym[gid] for gid in p1_liver_ids if gid in id_to_screen_sym])
    print(f"    -> Genes: {', '.join(p1_liver_syms)}")

    # Test: is interface over-represented in residual?
    interface_ids_res = symbols_to_geneids(interface_screen_symbols, sym2id) & residual_univ_set
    a_iface, _, _, _, odds_iface, p_iface = fisher_overrep(
        res_dep_set, interface_ids_res, residual_univ_set
    )
    print(
        f"  Interface over-rep in residual:      "
        f"OR={odds_iface:.3f}, p={p_iface:.2e}  "
        f"({a_iface}/{len(res_dep_set)} residual depleted are interface genes)"
    )
    p6_iface_ids = res_dep_set & interface_ids_res
    p6_iface_syms = sorted([id_to_screen_sym[gid] for gid in p6_iface_ids if gid in id_to_screen_sym])
    p6_iface_annotated = [f"{g} ({interface_subcats_sym.get(g, 'unknown')})" for g in p6_iface_syms]
    print(f"    -> Genes: {', '.join(p6_iface_annotated)}")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Figures  → {FIGS_DIR}/")
    print(f"  Tables   → {TABLES_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
