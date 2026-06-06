"""
Resume / Candidate Screening System
Future Interns - Machine Learning Task 3 (2026)
--------------------------------------------------
Uses NLP (spaCy, NLTK, scikit-learn) to:
  - Extract skills from resumes
  - Compare with a job description
  - Score and rank candidates
  - Identify skill gaps
"""

import re
import json
import warnings
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
#  NLTK downloads (run once)
# --------------------------------------------------------------------------- #
for pkg in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger",
            "punkt_tab"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
#  Skill master list  (extend as needed)
# --------------------------------------------------------------------------- #
SKILLS_DB = {
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "r", "scala",
    "go", "rust", "kotlin", "swift", "php", "ruby", "matlab",
    # ML / AI
    "machine learning", "deep learning", "neural network", "nlp",
    "natural language processing", "computer vision", "reinforcement learning",
    "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost", "lightgbm",
    "hugging face", "transformers", "bert", "gpt", "llm",
    # Data
    "pandas", "numpy", "matplotlib", "seaborn", "plotly", "tableau", "power bi",
    "sql", "mysql", "postgresql", "mongodb", "spark", "hadoop", "airflow",
    "data analysis", "data visualization", "statistics", "etl",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "git", "linux",
    "terraform", "jenkins", "mlops",
    # Web
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
    "html", "css", "rest api", "graphql",
    # Soft skills
    "communication", "teamwork", "leadership", "problem solving",
    "critical thinking", "time management", "agile", "scrum",
}


# --------------------------------------------------------------------------- #
#  Text Preprocessor
# --------------------------------------------------------------------------- #
class TextPreprocessor:
    """Clean and normalise raw text."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self._stop = set(stopwords.words("english"))

    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s\+\#\/\.]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        tokens = word_tokenize(self.clean(text))
        return [self.lemmatizer.lemmatize(t) for t in tokens
                if t not in self._stop and len(t) > 1]

    def process(self, text: str) -> str:
        return " ".join(self.tokenize(text))


# --------------------------------------------------------------------------- #
#  Skill Extractor
# --------------------------------------------------------------------------- #
class SkillExtractor:
    """Extract skills from text using keyword matching + spaCy NER."""

    def __init__(self, skills_db: set = None):
        self.skills_db = skills_db or SKILLS_DB
        # Build sorted list (longer phrases first to avoid partial matches)
        self._skills_sorted = sorted(self.skills_db, key=len, reverse=True)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except (OSError, Exception):
            self.nlp = None  # keyword matching is still fully functional

    def extract(self, text: str) -> List[str]:
        text_lower = text.lower()
        found = set()
        for skill in self._skills_sorted:
            pattern = r"\b" + re.escape(skill) + r"\b"
            if re.search(pattern, text_lower):
                found.add(skill)
        return sorted(found)


# --------------------------------------------------------------------------- #
#  Candidate
# --------------------------------------------------------------------------- #
class Candidate:
    def __init__(self, name: str, resume_text: str):
        self.name = name
        self.raw_text = resume_text
        self.skills: List[str] = []
        self.processed_text: str = ""
        self.score: float = 0.0
        self.rank: int = 0
        self.matched_skills: List[str] = []
        self.missing_skills: List[str] = []

    def to_dict(self) -> Dict:
        return {
            "Rank": self.rank,
            "Candidate": self.name,
            "Score (%)": round(self.score * 100, 2),
            "Skills Found": len(self.matched_skills),
            "Matched Skills": ", ".join(self.matched_skills),
            "Missing Skills": ", ".join(self.missing_skills),
        }


# --------------------------------------------------------------------------- #
#  Resume Screener  (main class)
# --------------------------------------------------------------------------- #
class ResumeScreener:
    """
    End-to-end resume screening pipeline.

    Usage
    -----
    screener = ResumeScreener()
    screener.set_job_description(jd_text)
    screener.add_candidate("Alice", alice_resume_text)
    screener.add_candidate("Bob",   bob_resume_text)
    results = screener.rank_candidates()
    screener.print_report()
    """

    def __init__(self, skill_weight: float = 0.5, tfidf_weight: float = 0.5):
        assert abs(skill_weight + tfidf_weight - 1.0) < 1e-6, \
            "Weights must sum to 1"
        self.skill_weight = skill_weight
        self.tfidf_weight = tfidf_weight

        self.preprocessor = TextPreprocessor()
        self.skill_extractor = SkillExtractor()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)

        self._jd_text: str = ""
        self._jd_processed: str = ""
        self._jd_skills: List[str] = []
        self.candidates: List[Candidate] = []

    # ---------------------------------------------------------------------- #
    def set_job_description(self, jd_text: str) -> None:
        self._jd_text = jd_text
        self._jd_processed = self.preprocessor.process(jd_text)
        self._jd_skills = self.skill_extractor.extract(jd_text)
        print(f"✅ Job description loaded. Required skills detected: "
              f"{len(self._jd_skills)}")
        if self._jd_skills:
            print(f"   Skills: {', '.join(self._jd_skills)}")

    # ---------------------------------------------------------------------- #
    def add_candidate(self, name: str, resume_text: str) -> None:
        c = Candidate(name, resume_text)
        c.processed_text = self.preprocessor.process(resume_text)
        c.skills = self.skill_extractor.extract(resume_text)
        self.candidates.append(c)

    # ---------------------------------------------------------------------- #
    def _tfidf_scores(self) -> np.ndarray:
        corpus = [self._jd_processed] + [c.processed_text
                                          for c in self.candidates]
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        jd_vec = tfidf_matrix[0]
        resume_vecs = tfidf_matrix[1:]
        similarities = cosine_similarity(jd_vec, resume_vecs).flatten()
        return similarities

    def _skill_scores(self) -> np.ndarray:
        if not self._jd_skills:
            return np.zeros(len(self.candidates))
        jd_skill_set = set(self._jd_skills)
        scores = []
        for c in self.candidates:
            matched = jd_skill_set.intersection(set(c.skills))
            score = len(matched) / len(jd_skill_set)
            scores.append(score)
        return np.array(scores)

    # ---------------------------------------------------------------------- #
    def rank_candidates(self) -> pd.DataFrame:
        if not self.candidates:
            raise ValueError("No candidates added.")
        if not self._jd_text:
            raise ValueError("No job description set.")

        tfidf_scores = self._tfidf_scores()
        skill_scores = self._skill_scores()
        combined = (self.tfidf_weight * tfidf_scores
                    + self.skill_weight * skill_scores)

        jd_skill_set = set(self._jd_skills)
        for i, c in enumerate(self.candidates):
            c.score = float(combined[i])
            c.matched_skills = sorted(
                jd_skill_set.intersection(set(c.skills)))
            c.missing_skills = sorted(
                jd_skill_set.difference(set(c.skills)))

        # Sort descending
        self.candidates.sort(key=lambda x: x.score, reverse=True)
        for rank, c in enumerate(self.candidates, 1):
            c.rank = rank

        rows = [c.to_dict() for c in self.candidates]
        return pd.DataFrame(rows)

    # ---------------------------------------------------------------------- #
    def print_report(self) -> None:
        if not self.candidates:
            print("No candidates to report.")
            return

        print("\n" + "=" * 65)
        print("         📋  RESUME SCREENING REPORT")
        print("=" * 65)
        print(f"Job Required Skills ({len(self._jd_skills)}): "
              f"{', '.join(self._jd_skills) if self._jd_skills else 'N/A'}\n")

        for c in self.candidates:
            bar_len = int(c.score * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"#{c.rank:2d}  {c.name:<20s}  [{bar}]  "
                  f"{c.score * 100:5.1f}%")
            print(f"     ✅ Matched ({len(c.matched_skills)}): "
                  f"{', '.join(c.matched_skills) or '—'}")
            print(f"     ❌ Missing ({len(c.missing_skills)}): "
                  f"{', '.join(c.missing_skills) or '—'}")
            print()

        print("=" * 65)
        top = self.candidates[0]
        print(f"🏆  Top Candidate: {top.name}  |  Score: "
              f"{top.score * 100:.1f}%")
        print("=" * 65)

    # ---------------------------------------------------------------------- #
    def save_results(self, output_path: str = "outputs/results.csv") -> None:
        df = pd.DataFrame([c.to_dict() for c in self.candidates])
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"💾  Results saved → {output_path}")