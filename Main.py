"""
main.py – Run the Resume Screening System
==========================================
Usage:
    python main.py                      # uses built-in sample data
    python main.py --role ml_engineer   # choose job role
    python main.py --csv path/to/resumes.csv  # load from CSV

CLI Options:
    --role    data_scientist | ml_engineer | frontend_developer  (default: data_scientist)
    --csv     path to a CSV with columns: name, resume_text
    --output  path for results CSV  (default: outputs/results.csv)
    --skill-weight   float 0–1  (default: 0.5)
    --tfidf-weight   float 0–1  (default: 0.5)
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from resume_screener import ResumeScreener
from sample_data import SAMPLE_RESUMES, JOB_DESCRIPTIONS
import pandas as pd


# ──────────────────────────────────────────────────────────────
def load_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    assert "name" in df.columns and "resume_text" in df.columns, \
        "CSV must have columns: name, resume_text"
    return dict(zip(df["name"], df["resume_text"]))


# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Resume / Candidate Screening System – Future Interns ML Task 3"
    )
    parser.add_argument("--role", default="data_scientist",
                        choices=list(JOB_DESCRIPTIONS.keys()),
                        help="Job role to screen for")
    parser.add_argument("--csv", default=None,
                        help="Path to CSV file with columns: name, resume_text")
    parser.add_argument("--output", default="outputs/results.csv",
                        help="Path for output CSV results")
    parser.add_argument("--skill-weight", type=float, default=0.5,
                        help="Weight for skill-matching score (0–1)")
    parser.add_argument("--tfidf-weight", type=float, default=0.5,
                        help="Weight for TF-IDF similarity score (0–1)")
    args = parser.parse_args()

    print("\n╔══════════════════════════════════════════════╗")
    print("║  Resume / Candidate Screening System         ║")
    print("║  Future Interns – ML Task 3 (2026)           ║")
    print("╚══════════════════════════════════════════════╝\n")

    # ------------------------------------------------------------------
    # 1. Initialise screener
    # ------------------------------------------------------------------
    screener = ResumeScreener(
        skill_weight=args.skill_weight,
        tfidf_weight=args.tfidf_weight,
    )

    # ------------------------------------------------------------------
    # 2. Set job description
    # ------------------------------------------------------------------
    jd = JOB_DESCRIPTIONS[args.role]
    print(f"📌 Selected Role : {args.role.replace('_', ' ').title()}")
    screener.set_job_description(jd)

    # ------------------------------------------------------------------
    # 3. Load candidates
    # ------------------------------------------------------------------
    if args.csv:
        candidates = load_from_csv(args.csv)
        print(f"\n📂 Loaded {len(candidates)} candidates from {args.csv}")
    else:
        candidates = SAMPLE_RESUMES
        print(f"\n📂 Using {len(candidates)} built-in sample candidates")

    for name, text in candidates.items():
        screener.add_candidate(name, text)

    # ------------------------------------------------------------------
    # 4. Rank and report
    # ------------------------------------------------------------------
    print("\n🔄 Running screening pipeline …")
    results_df = screener.rank_candidates()

    screener.print_report()

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    screener.save_results(args.output)

    print("\n📊 Results Table:\n")
    print(results_df.to_string(index=False))
    print()


if __name__ == "__main__":
    main()