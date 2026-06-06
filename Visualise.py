"""
visualise.py – Generate visual comparison charts for screened candidates.
Run after main.py or import and call directly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib
matplotlib.use("Agg")   # headless (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from resume_screener import ResumeScreener
from sample_data import SAMPLE_RESUMES, JOB_DESCRIPTIONS


# ──────────────────────────────────────────────────────────────
def plot_candidate_scores(screener: ResumeScreener,
                          save_path: str = "outputs/candidate_scores.png"):
    candidates = screener.candidates
    names  = [c.name for c in candidates]
    scores = [c.score * 100 for c in candidates]
    skill_pct   = [len(c.matched_skills) / max(len(screener._jd_skills), 1) * 100
                   for c in candidates]
    tfidf_pct   = [(c.score - screener.skill_weight *
                    len(c.matched_skills) / max(len(screener._jd_skills), 1))
                   / screener.tfidf_weight * 100
                   for c in candidates]

    x = np.arange(len(names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Resume Screening – Candidate Comparison", fontsize=14,
                 fontweight="bold")

    # --- Bar chart: Overall score ---
    colours = ["#2ecc71" if s == max(scores) else "#3498db" for s in scores]
    axes[0].bar(x, scores, color=colours, edgecolor="white", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    axes[0].set_ylabel("Score (%)")
    axes[0].set_title("Overall Fit Score")
    axes[0].set_ylim(0, 105)
    axes[0].axhline(y=np.mean(scores), color="red", linestyle="--",
                    linewidth=1, label="Average")
    for i, v in enumerate(scores):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=8)
    axes[0].legend(fontsize=8)

    # --- Grouped bar: Skill match vs TF-IDF similarity ---
    b1 = axes[1].bar(x - width/2, skill_pct, width, label="Skill Match %",
                     color="#e67e22", edgecolor="white")
    b2 = axes[1].bar(x + width/2, tfidf_pct, width, label="TF-IDF Similarity %",
                     color="#9b59b6", edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    axes[1].set_ylabel("Score (%)")
    axes[1].set_title("Score Breakdown")
    axes[1].set_ylim(0, 115)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"📊  Score comparison chart saved → {save_path}")


# ──────────────────────────────────────────────────────────────
def plot_skill_heatmap(screener: ResumeScreener,
                       save_path: str = "outputs/skill_heatmap.png"):
    jd_skills = screener._jd_skills
    if not jd_skills:
        print("⚠️  No required skills in JD – skipping heatmap.")
        return

    candidates = screener.candidates
    matrix = []
    for c in candidates:
        row = [1 if s in c.skills else 0 for s in jd_skills]
        matrix.append(row)

    mat = np.array(matrix)
    fig, ax = plt.subplots(figsize=(max(8, len(jd_skills) * 0.7),
                                    max(4, len(candidates) * 0.5)))
    im = ax.imshow(mat, cmap="YlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(jd_skills)))
    ax.set_yticks(np.arange(len(candidates)))
    ax.set_xticklabels(jd_skills, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([c.name for c in candidates], fontsize=9)

    for i in range(len(candidates)):
        for j in range(len(jd_skills)):
            ax.text(j, i, "✓" if mat[i, j] else "✗",
                    ha="center", va="center", fontsize=10,
                    color="white" if mat[i, j] else "#c0392b")

    ax.set_title("Skill Coverage Heatmap\n(✓ = skill present, ✗ = missing)",
                 fontsize=12)
    plt.colorbar(im, ax=ax, label="Has Skill")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"🗺️   Skill heatmap saved → {save_path}")


# ──────────────────────────────────────────────────────────────
def plot_skill_gap(screener: ResumeScreener,
                   save_path: str = "outputs/skill_gap.png"):
    """Show how many candidates have each required skill."""
    jd_skills = screener._jd_skills
    if not jd_skills:
        return
    counts = {s: sum(1 for c in screener.candidates if s in c.skills)
              for s in jd_skills}
    total = len(screener.candidates)
    skills_sorted = sorted(counts, key=counts.get)
    vals = [counts[s] / total * 100 for s in skills_sorted]

    fig, ax = plt.subplots(figsize=(8, max(4, len(skills_sorted) * 0.4)))
    colours = ["#27ae60" if v >= 60 else "#e67e22" if v >= 30 else "#e74c3c"
               for v in vals]
    bars = ax.barh(skills_sorted, vals, color=colours, edgecolor="white")
    ax.set_xlabel("% of Candidates Having Skill")
    ax.set_title("Skill Gap Analysis\n(red = rare skill, green = common skill)")
    ax.set_xlim(0, 110)
    for bar, val in zip(bars, vals):
        ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", fontsize=8)

    legend_patches = [
        mpatches.Patch(color="#27ae60", label="≥60% candidates"),
        mpatches.Patch(color="#e67e22", label="30–59%"),
        mpatches.Patch(color="#e74c3c", label="<30% (skill gap)"),
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"📉  Skill gap chart saved → {save_path}")


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    screener = ResumeScreener()
    screener.set_job_description(JOB_DESCRIPTIONS["data_scientist"])
    for name, text in SAMPLE_RESUMES.items():
        screener.add_candidate(name, text)
    screener.rank_candidates()

    plot_candidate_scores(screener)
    plot_skill_heatmap(screener)
    plot_skill_gap(screener)
    print("\n✅  All visualisations saved to outputs/")