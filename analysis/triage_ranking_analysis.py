"""
Triage Ranking & Calibration Analysis for RadTriage AI
=======================================================
KEY FINDING: The original urgency score cap at 1.0 destroys all ranking signal
because 99.97% of cases (soft sigmoid probabilities) exceed the cap.
Raw (uncapped) urgency scores range 0.70–5.62 and carry rich ranking signal.

This script:
  1. Shows score distributions per true tier (using raw scores)
  2. Computes AUROC of raw urgency score as Emergent / Urgent+ detector
  3. Top-K Emergent recall curve (AI vs random)
  4. Calibrated threshold search (matching ground-truth tier proportions)
  5. Per-class contribution to urgency score (sensitivity analysis)

Outputs saved to analysis/outputs/
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support

# ── Paths ──────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parent.parent
PREDS    = BASE / "kaggle/kaggle_run_1/test_predictions_full.csv"
OUT_DIR  = BASE / "analysis/outputs"
SUM_FILE = OUT_DIR / "triage_summary.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
TIER_ORDER = ["Emergent", "Urgent", "Routine"]

def raw_urgency(row, prefix):
    """Raw (uncapped) weighted sum of probabilities."""
    return sum(float(row[f"{prefix}_{c}"]) * SEVERITY_WEIGHTS[c] for c in CLASSES)

def apply_overrides(row, prefix, fallback):
    if float(row[f"{prefix}_Pneumothorax"]) >= 0.60:
        return "Emergent"
    if float(row[f"{prefix}_Edema"]) >= 0.70 or float(row[f"{prefix}_Pneumonia"]) >= 0.70:
        return "Urgent"
    return fallback

def assign_tier_raw(raw_score, emergent_thr, urgent_thr):
    if raw_score >= emergent_thr: return "Emergent"
    if raw_score >= urgent_thr:   return "Urgent"
    return "Routine"

# ── Load ───────────────────────────────────────────────────────────────────
print("Loading …")
df = pd.read_csv(PREDS)
print(f"  {len(df):,} rows")

# Raw uncapped urgency scores
df["pred_raw"] = df.apply(lambda r: raw_urgency(r, "prob"), axis=1)
df["true_raw"] = df.apply(lambda r: raw_urgency(r, "true"), axis=1)

# True tier (binary labels, original thresholds work fine since binary scores are ~0-3 max)
def row_true_tier(r):
    s = r["true_raw"]
    if s >= 0.75:   t = "Emergent"
    elif s >= 0.40: t = "Urgent"
    else:           t = "Routine"
    return apply_overrides(r, "true", t)

df["true_tier"] = df.apply(row_true_tier, axis=1)
true_dist = df["true_tier"].value_counts().reindex(TIER_ORDER, fill_value=0)

print(f"\n  True tier distribution:")
for t in TIER_ORDER:
    print(f"    {t:<10}: {true_dist[t]:>6,}  ({true_dist[t]/len(df)*100:.1f}%)")

# ── 1. Raw score stats per true tier ─────────────────────────────────────
print("\n── Predicted Raw Urgency Score by True Tier ─────────────")
for tier in TIER_ORDER:
    sub = df[df["true_tier"] == tier]["pred_raw"]
    print(f"  {tier:<10}  n={len(sub):>6,}  mean={sub.mean():.3f}  "
          f"median={sub.median():.3f}  p25={sub.quantile(.25):.3f}  p75={sub.quantile(.75):.3f}")

# ── 2. Calibrated thresholds (percentile-based) ───────────────────────────
target_emg_pct  = true_dist["Emergent"] / len(df)
target_urg_pct  = true_dist["Urgent"]   / len(df)

cal_emg_thr = df["pred_raw"].quantile(1 - target_emg_pct)
cal_urg_thr = df["pred_raw"].quantile(1 - target_emg_pct - target_urg_pct)

print(f"\n── Calibrated Thresholds ────────────────────────────────")
print(f"  Target Emergent: {target_emg_pct*100:.1f}%  →  raw score ≥ {cal_emg_thr:.3f}")
print(f"  Target Urgent:   {target_urg_pct*100:.1f}%  →  raw score ≥ {cal_urg_thr:.3f}")
print(f"  (Original thresholds 0.75 / 0.40 designed for binary inputs, not sigmoid probabilities)")

# Assign calibrated tiers (with overrides)
def assign_calibrated_tier(r):
    t = assign_tier_raw(r["pred_raw"], cal_emg_thr, cal_urg_thr)
    return apply_overrides(r, "prob", t)

df["pred_tier_cal"] = df.apply(assign_calibrated_tier, axis=1)

cal_acc = (df["pred_tier_cal"] == df["true_tier"]).mean()
prec, rec, f1, sup = precision_recall_fscore_support(
    df["true_tier"], df["pred_tier_cal"], labels=TIER_ORDER, zero_division=0
)

print(f"\n── Calibrated Tier Performance ──────────────────────────")
print(f"  Overall accuracy: {cal_acc*100:.2f}%")
print(f"  {'Tier':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>10}")
for i, t in enumerate(TIER_ORDER):
    print(f"  {t:<12} {prec[i]:>10.3f} {rec[i]:>8.3f} {f1[i]:>8.3f} {sup[i]:>10,}")

cal_dist = df["pred_tier_cal"].value_counts().reindex(TIER_ORDER, fill_value=0)
print(f"\n  Calibrated distribution: {dict(cal_dist)}")

# ── 3. AUROC: raw urgency score as Emergent/Urgent+ detector ─────────────
df["is_emergent"]    = (df["true_tier"] == "Emergent").astype(int)
df["is_urgent_plus"] = (df["true_tier"].isin(["Emergent","Urgent"])).astype(int)

auc_emg = roc_auc_score(df["is_emergent"],    df["pred_raw"])
auc_urg = roc_auc_score(df["is_urgent_plus"],  df["pred_raw"])

print(f"\n── Ranking Quality (AUROC of raw urgency score) ─────────")
print(f"  Emergent vs Urgent+Routine : {auc_emg:.4f}")
print(f"  Emergent+Urgent vs Routine : {auc_urg:.4f}")

# ── 4. Top-K Emergent recall ──────────────────────────────────────────────
total_emergent = df["is_emergent"].sum()
n_total = len(df)
df["pred_rank"] = df["pred_raw"].rank(method="first", ascending=False).astype(int)

k_percents = [5, 10, 15, 20, 25, 30, 40, 50]
topk_recalls = {}
print(f"\n── Top-K% Emergent Recall  (total Emergent = {total_emergent:,}) ──")
print(f"  {'Top K%':>8}  {'Cases reviewed':>16}  {'Emergent found':>16}  {'Recall':>8}  {'vs Random':>12}")
for k in k_percents:
    k_n = int(n_total * k / 100)
    found = (df[df["pred_rank"] <= k_n]["is_emergent"]).sum()
    recall = found / total_emergent
    random_recall = k / 100
    lift = recall / random_recall
    topk_recalls[k] = float(recall)
    print(f"  {k:>7}%  {k_n:>16,}  {found:>16,}  {recall*100:>7.1f}%  {lift:>8.2f}x lift")

# ── 5. Worklist simulation: time-to-first-critical ────────────────────────
print(f"\n── Full Worklist Re-Rank (all {n_total:,} cases) ────────────")
# Median position of Emergent cases
emg_ai_ranks     = df[df["is_emergent"]==1]["pred_rank"]
emg_random_ranks = df[df["is_emergent"]==1].index + 1  # random = original row order

ai_median_rank     = emg_ai_ranks.median()
random_median_rank = n_total / 2  # expected median under random

print(f"  Median AI rank of Emergent cases    : {ai_median_rank:.0f} / {n_total:,}")
print(f"  Expected median rank (random order) : {random_median_rank:.0f} / {n_total:,}")
print(f"  AI brings Emergent cases {random_median_rank/ai_median_rank:.1f}x earlier on average")

# Cases reviewed before capturing 90% of Emergent
target_90 = 0.90 * total_emergent
for k in range(1, n_total+1):
    found = (df[df["pred_rank"] <= k]["is_emergent"]).sum()
    if found >= target_90:
        pct_reviewed = k / n_total * 100
        print(f"  To find 90% of Emergent: review top {k:,} cases ({pct_reviewed:.1f}%) with AI")
        print(f"  vs ~{int(0.90 * n_total):,} cases ({90:.1f}%) in random order")
        break

# ── 6. Plots ───────────────────────────────────────────────────────────────

# 6a. Raw score distributions per tier (violin)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Box plot of raw scores
data_by_tier = [df[df["true_tier"]==t]["pred_raw"].values for t in TIER_ORDER]
colors_tier  = ["#d9534f", "#f0ad4e", "#5cb85c"]
bp = axes[0].boxplot(data_by_tier, tick_labels=TIER_ORDER, patch_artist=True,
                     medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], colors_tier):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].axhline(cal_emg_thr, color="#8B0000", linestyle="-.", linewidth=1.4,
                label=f"Calibrated Emergent thr ({cal_emg_thr:.2f})")
axes[0].axhline(cal_urg_thr, color="#8B4513", linestyle="-.", linewidth=1.4,
                label=f"Calibrated Urgent thr ({cal_urg_thr:.2f})")
axes[0].axhline(0.75, color="#d9534f", linestyle="--", linewidth=0.9, alpha=0.5,
                label="Original thr 0.75 (designed for binary)")
axes[0].set_xlabel("True Triage Tier", fontsize=11)
axes[0].set_ylabel("Raw Urgency Score (uncapped)", fontsize=11)
axes[0].set_title("Urgency Score Distribution\nby True Tier", fontsize=11, fontweight="bold")
axes[0].legend(fontsize=7.5)

# ROC curve
fpr, tpr, _ = roc_curve(df["is_emergent"], df["pred_raw"])
axes[1].plot(fpr, tpr, color="#4C72B0", linewidth=2, label=f"AUC = {auc_emg:.3f}")
axes[1].plot([0,1],[0,1], "k--", linewidth=0.8, label="Random (AUC=0.500)")
axes[1].set_xlabel("False Positive Rate", fontsize=11)
axes[1].set_ylabel("True Positive Rate", fontsize=11)
axes[1].set_title("ROC – Raw Urgency Score\nEmergent vs Urgent+Routine", fontsize=11, fontweight="bold")
axes[1].legend(fontsize=10)

plt.suptitle("RadTriage AI – Ranking Quality Analysis", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(OUT_DIR / "score_dist_and_roc.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n[saved] score_dist_and_roc.png")

# 6b. Top-K recall curve
ks_dense = list(range(1, 101))
recalls_dense = []
for k in ks_dense:
    k_n = int(n_total * k / 100)
    found = (df[df["pred_rank"] <= k_n]["is_emergent"]).sum()
    recalls_dense.append(found / total_emergent)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(ks_dense, [r*100 for r in recalls_dense], color="#4C72B0", linewidth=2.5,
        label=f"AI-ranked order (AUROC={auc_emg:.3f})")
ax.plot(ks_dense, ks_dense, "k--", linewidth=1.2, label="Random order (baseline)")
ax.fill_between(ks_dense, ks_dense, [r*100 for r in recalls_dense],
                alpha=0.15, color="#4C72B0", label="AI advantage")
for k, pct in [(10, recalls_dense[9]*100), (20, recalls_dense[19]*100), (30, recalls_dense[29]*100)]:
    ax.annotate(f"{pct:.0f}% found\n@ top {k}%", xy=(k, pct),
                xytext=(k+6, pct-12), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", lw=0.9),
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", lw=0.6))
ax.set_xlabel("% of Worklist Reviewed", fontsize=12)
ax.set_ylabel("% of Emergent Cases Found", fontsize=12)
ax.set_title("Emergent Case Detection: AI-Ranked vs Random Order\n(25,596 test cases)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.set_xlim(0, 100); ax.set_ylim(0, 105)
plt.tight_layout()
fig.savefig(OUT_DIR / "topk_emergent_recall.png", dpi=150)
plt.close(fig)
print(f"[saved] topk_emergent_recall.png")

# 6c. Calibrated confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm_cal = confusion_matrix(df["true_tier"], df["pred_tier_cal"], labels=TIER_ORDER)
fig, ax = plt.subplots(figsize=(7, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_cal, display_labels=TIER_ORDER)
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title(f"Triage Confusion Matrix – Calibrated Thresholds\n"
             f"Overall Accuracy: {cal_acc*100:.1f}%  |  Emergent Recall: {rec[0]*100:.1f}%",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Predicted Tier", fontsize=11)
ax.set_ylabel("True Tier", fontsize=11)
plt.tight_layout()
fig.savefig(OUT_DIR / "triage_confusion_matrix_calibrated.png", dpi=150)
plt.close(fig)
print(f"[saved] triage_confusion_matrix_calibrated.png")

# ── 7. Update summary JSON ─────────────────────────────────────────────────
with open(SUM_FILE) as f:
    summary = json.load(f)

summary.update({
    "raw_score_stats": {
        "min": round(df["pred_raw"].min(), 3),
        "max": round(df["pred_raw"].max(), 3),
        "mean": round(df["pred_raw"].mean(), 3),
        "std": round(df["pred_raw"].std(), 3),
        "pct_exceed_original_cap_1": round(float((df["pred_raw"] > 1.0).mean()), 4),
    },
    "calibration_issue": "Original thresholds (0.75/0.40) designed for binary inputs. 99.97% of sigmoid-based scores exceed cap of 1.0, destroying ranking signal.",
    "calibrated_thresholds": {
        "emergent_thr": round(float(cal_emg_thr), 3),
        "urgent_thr":   round(float(cal_urg_thr), 3),
    },
    "ranking_auroc_emergent_vs_rest":      round(float(auc_emg), 4),
    "ranking_auroc_urgentplus_vs_routine": round(float(auc_urg), 4),
    "topk_emergent_recall": {f"top{k}pct": round(topk_recalls[k], 4) for k in k_percents},
    "calibrated_tier_performance": {
        "overall_accuracy":    round(float(cal_acc), 4),
        "emergent_precision":  round(float(prec[0]), 4),
        "emergent_recall":     round(float(rec[0]), 4),
        "emergent_f1":         round(float(f1[0]), 4),
        "urgent_precision":    round(float(prec[1]), 4),
        "urgent_recall":       round(float(rec[1]), 4),
        "urgent_f1":           round(float(f1[1]), 4),
        "routine_precision":   round(float(prec[2]), 4),
        "routine_recall":      round(float(rec[2]), 4),
        "routine_f1":          round(float(f1[2]), 4),
    },
    "worklist_simulation": {
        "median_ai_rank_emergent":     int(ai_median_rank),
        "median_random_rank_emergent": int(random_median_rank),
        "speedup_factor":              round(random_median_rank / ai_median_rank, 2),
    },
})

with open(SUM_FILE, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n[updated] triage_summary.json")
print("\n✓ Ranking analysis complete.")
