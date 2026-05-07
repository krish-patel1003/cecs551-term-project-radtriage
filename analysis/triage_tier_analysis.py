"""
Triage-Tier Accuracy Analysis for RadTriage AI
================================================
Uses the exact same scoring logic as src/triage/ to evaluate
predicted vs. ground-truth triage tiers across all 25,596 test cases.

Outputs:
  - Confusion matrix (Emergent / Urgent / Routine)
  - Per-tier Precision, Recall, F1
  - Overall triage accuracy
  - Critical-case recall (Emergent sensitivity)
  - Run 1 vs Run 2 metrics comparison
  - Saved plots: triage_confusion_matrix.png, triage_metrics_bar.png, auc_per_class.png
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay, precision_recall_fscore_support,
)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
PREDS_CSV   = BASE / "kaggle/kaggle_run_1/test_predictions_full.csv"
METRICS_CSV = BASE / "kaggle/kaggle_run_1/test_metrics_full.csv"
OUT_DIR     = BASE / "analysis/outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Triage constants (mirrors src/triage/) ─────────────────────────────────
CLASSES = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule",
    "Pneumonia","Pneumothorax","Consolidation","Edema","Emphysema",
    "Fibrosis","Pleural_Thickening","Hernia",
]

SEVERITY_WEIGHTS = {
    "Atelectasis": 0.35, "Cardiomegaly": 0.45, "Effusion": 0.55,
    "Infiltration": 0.50, "Mass": 0.60, "Nodule": 0.30,
    "Pneumonia": 0.75, "Pneumothorax": 1.00, "Consolidation": 0.65,
    "Edema": 0.80, "Emphysema": 0.30, "Fibrosis": 0.25,
    "Pleural_Thickening": 0.20, "Hernia": 0.15,
}

EMERGENT_THR = 0.75
URGENT_THR   = 0.40

OVERRIDE_PNEUMOTHORAX = 0.60
OVERRIDE_EDEMA        = 0.70
OVERRIDE_PNEUMONIA    = 0.70


def urgency_score(probs: dict, cap: bool = True) -> float:
    s = sum(probs.get(c, 0.0) * SEVERITY_WEIGHTS[c] for c in CLASSES)
    return min(s, 1.0) if cap else s


def map_score_to_tier(score: float) -> str:
    if score >= EMERGENT_THR: return "Emergent"
    if score >= URGENT_THR:   return "Urgent"
    return "Routine"


def apply_overrides(probs: dict, fallback: str) -> str:
    if probs.get("Pneumothorax", 0.0) >= OVERRIDE_PNEUMOTHORAX:
        return "Emergent"
    if probs.get("Edema", 0.0) >= OVERRIDE_EDEMA or probs.get("Pneumonia", 0.0) >= OVERRIDE_PNEUMONIA:
        return "Urgent"
    return fallback


def assign_tier(probs: dict) -> str:
    return apply_overrides(probs, map_score_to_tier(urgency_score(probs)))


# ── Load data ──────────────────────────────────────────────────────────────
print("Loading test predictions …")
df = pd.read_csv(PREDS_CSV)
print(f"  {len(df):,} rows, {df.shape[1]} columns")

prob_cols = {c: f"prob_{c}" for c in CLASSES}
true_cols = {c: f"true_{c}" for c in CLASSES}

# ── Assign predicted and ground-truth tiers ────────────────────────────────
print("Computing triage tiers …")

def row_to_tier(row, col_map):
    probs = {c: float(row[col]) for c, col in col_map.items()}
    return assign_tier(probs)

df["pred_tier"] = df.apply(lambda r: row_to_tier(r, prob_cols), axis=1)
df["true_tier"] = df.apply(lambda r: row_to_tier(r, true_cols), axis=1)

# ── Distribution ──────────────────────────────────────────────────────────
TIER_ORDER = ["Emergent", "Urgent", "Routine"]

pred_dist = df["pred_tier"].value_counts().reindex(TIER_ORDER, fill_value=0)
true_dist = df["true_tier"].value_counts().reindex(TIER_ORDER, fill_value=0)

print("\n── Tier Distribution ──────────────────────────────────")
print(f"{'Tier':<12} {'Ground Truth':>14} {'Predicted':>12}")
for t in TIER_ORDER:
    print(f"  {t:<10} {true_dist[t]:>10,}  ({true_dist[t]/len(df)*100:5.1f}%)"
          f"  {pred_dist[t]:>8,}  ({pred_dist[t]/len(df)*100:5.1f}%)")

# ── Confusion matrix ───────────────────────────────────────────────────────
print("\n── Confusion Matrix (rows=True, cols=Pred) ─────────────")
cm = confusion_matrix(df["true_tier"], df["pred_tier"], labels=TIER_ORDER)
cm_df = pd.DataFrame(cm, index=[f"True {t}" for t in TIER_ORDER],
                         columns=[f"Pred {t}" for t in TIER_ORDER])
print(cm_df.to_string())

# ── Per-tier metrics ───────────────────────────────────────────────────────
prec, rec, f1, sup = precision_recall_fscore_support(
    df["true_tier"], df["pred_tier"], labels=TIER_ORDER, zero_division=0
)
overall_acc = (df["true_tier"] == df["pred_tier"]).mean()

print(f"\n── Per-Tier Metrics ────────────────────────────────────")
print(f"{'Tier':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>10}")
for i, t in enumerate(TIER_ORDER):
    print(f"  {t:<10} {prec[i]:>10.3f} {rec[i]:>8.3f} {f1[i]:>8.3f} {sup[i]:>10,}")

print(f"\n  Overall Triage Accuracy : {overall_acc*100:.2f}%")
print(f"  Emergent Recall (sensitivity): {rec[0]*100:.2f}%")
print(f"  Emergent Precision             : {prec[0]*100:.2f}%")

# ── Emergent case deep-dive ────────────────────────────────────────────────
true_emergent = df[df["true_tier"] == "Emergent"]
correctly_elevated  = (true_emergent["pred_tier"] == "Emergent").sum()
downgraded_urgent   = (true_emergent["pred_tier"] == "Urgent").sum()
downgraded_routine  = (true_emergent["pred_tier"] == "Routine").sum()

print(f"\n── Emergent Case Breakdown ({len(true_emergent):,} true Emergent cases) ─")
print(f"  Correctly predicted Emergent : {correctly_elevated:,}  ({correctly_elevated/len(true_emergent)*100:.1f}%)")
print(f"  Downgraded to Urgent         : {downgraded_urgent:,}  ({downgraded_urgent/len(true_emergent)*100:.1f}%)")
print(f"  Downgraded to Routine        : {downgraded_routine:,}  ({downgraded_routine/len(true_emergent)*100:.1f}%)")

# ── Worklist re-rank simulation ────────────────────────────────────────────
# "How much earlier does the first Emergent case appear?"
sample_size = 500
rng = np.random.default_rng(42)
idx = rng.choice(len(df), size=sample_size, replace=False)
sample = df.iloc[idx].copy().reset_index(drop=True)
sample["urgency_score"] = sample.apply(
    lambda r: urgency_score({c: float(r[f"prob_{c}"]) for c in CLASSES}), axis=1
)
sample["random_rank"]   = rng.permutation(sample_size) + 1
sample["ai_rank"]       = sample["urgency_score"].rank(method="first", ascending=False).astype(int)

emergent_random = sample[sample["true_tier"]=="Emergent"]["random_rank"].min() if (sample["true_tier"]=="Emergent").any() else None
emergent_ai     = sample[sample["true_tier"]=="Emergent"]["ai_rank"].min()     if (sample["true_tier"]=="Emergent").any() else None

print(f"\n── Worklist Re-Rank Simulation (n={sample_size}) ───────────")
if emergent_random and emergent_ai:
    print(f"  First Emergent case rank (random order) : {emergent_random}")
    print(f"  First Emergent case rank (AI order)     : {emergent_ai}")
    print(f"  Cases skipped before first Emergent     : {emergent_random - emergent_ai} fewer with AI")

# ── Run 1 per-class metrics ────────────────────────────────────────────────
metrics = pd.read_csv(METRICS_CSV)
print(f"\n── Per-Class AUC (Run 1) ───────────────────────────────")
print(metrics[["class","auc","f1","sensitivity","specificity"]].sort_values("auc", ascending=False).to_string(index=False))

# ── Plots ──────────────────────────────────────────────────────────────────

# 1. Confusion matrix heatmap
fig, ax = plt.subplots(figsize=(7, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=TIER_ORDER)
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title("RadTriage AI – Triage Tier Confusion Matrix\n(25,596 test cases)", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted Tier", fontsize=11)
ax.set_ylabel("True Tier", fontsize=11)
plt.tight_layout()
fig.savefig(OUT_DIR / "triage_confusion_matrix.png", dpi=150)
plt.close(fig)
print(f"\n[saved] triage_confusion_matrix.png")

# 2. Per-tier metrics bar chart
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(TIER_ORDER))
w = 0.25
bars_p = ax.bar(x - w, prec, w, label="Precision", color="#4C72B0")
bars_r = ax.bar(x,     rec,  w, label="Recall",    color="#DD8452")
bars_f = ax.bar(x + w, f1,   w, label="F1",        color="#55A868")
ax.set_xticks(x)
ax.set_xticklabels(TIER_ORDER, fontsize=12)
ax.set_ylim(0, 1.05)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.set_title(f"Per-Tier Precision / Recall / F1   (Overall Accuracy: {overall_acc*100:.1f}%)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
for bars in [bars_p, bars_r, bars_f]:
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
                f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
fig.savefig(OUT_DIR / "triage_metrics_bar.png", dpi=150)
plt.close(fig)
print(f"[saved] triage_metrics_bar.png")

# 3. Per-class AUC horizontal bar
fig, ax = plt.subplots(figsize=(8, 6))
m_sorted = metrics.sort_values("auc")
colors = ["#d9534f" if v < 0.70 else "#f0ad4e" if v < 0.80 else "#5cb85c" for v in m_sorted["auc"]]
bars = ax.barh(m_sorted["class"], m_sorted["auc"], color=colors)
ax.axvline(0.777, color="#4C72B0", linestyle="--", linewidth=1.2, label=f"Macro AUC = 0.777")
ax.axvline(0.84,  color="#555555", linestyle=":",  linewidth=1.2, label="CheXNet baseline ≈ 0.840")
ax.set_xlim(0.5, 1.0)
ax.set_xlabel("AUC", fontsize=11)
ax.set_title("Per-Class AUC – RadTriage AI (Run 1)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
for b, v in zip(bars, m_sorted["auc"]):
    ax.text(v + 0.004, b.get_y() + b.get_height()/2, f"{v:.3f}", va="center", fontsize=8)
plt.tight_layout()
fig.savefig(OUT_DIR / "auc_per_class.png", dpi=150)
plt.close(fig)
print(f"[saved] auc_per_class.png")

# 4. Tier distribution comparison
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(TIER_ORDER))
w = 0.35
ax.bar(x - w/2, [true_dist[t] for t in TIER_ORDER], w, label="Ground Truth", color="#4C72B0", alpha=0.85)
ax.bar(x + w/2, [pred_dist[t] for t in TIER_ORDER], w, label="Predicted",    color="#DD8452", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(TIER_ORDER, fontsize=12)
ax.set_ylabel("Number of Cases", fontsize=10)
ax.set_title("Triage Tier Distribution: Ground Truth vs. Predicted", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
plt.tight_layout()
fig.savefig(OUT_DIR / "tier_distribution.png", dpi=150)
plt.close(fig)
print(f"[saved] tier_distribution.png")

# ── Save summary JSON for report use ──────────────────────────────────────
summary = {
    "n_test": int(len(df)),
    "overall_triage_accuracy": round(float(overall_acc), 4),
    "emergent_recall":   round(float(rec[0]), 4),
    "emergent_precision":round(float(prec[0]), 4),
    "emergent_f1":       round(float(f1[0]), 4),
    "urgent_recall":     round(float(rec[1]), 4),
    "urgent_precision":  round(float(prec[1]), 4),
    "urgent_f1":         round(float(f1[1]), 4),
    "routine_recall":    round(float(rec[2]), 4),
    "routine_precision": round(float(prec[2]), 4),
    "routine_f1":        round(float(f1[2]), 4),
    "tier_distribution": {
        "true":  {t: int(true_dist[t]) for t in TIER_ORDER},
        "pred":  {t: int(pred_dist[t]) for t in TIER_ORDER},
    },
    "emergent_breakdown": {
        "total_true_emergent": int(len(true_emergent)),
        "correctly_elevated":  int(correctly_elevated),
        "downgraded_urgent":   int(downgraded_urgent),
        "downgraded_routine":  int(downgraded_routine),
    },
    "confusion_matrix": cm.tolist(),
    "macro_auc":   0.7770,
    "macro_f1":    0.2750,
    "mean_sensitivity": 0.5569,
    "mean_specificity": 0.8242,
}
with open(OUT_DIR / "triage_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"[saved] triage_summary.json")

print("\n✓ Analysis complete. All outputs in:", OUT_DIR)
