from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class RetrievalResult:
    snippet: str
    score: float


@dataclass
class AnalysisResult:
    test_name: str
    assumptions: List[str]
    statistic_label: str
    statistic_value: float
    p_value: float
    effect_size_label: str
    effect_size_value: float
    confidence_interval: Tuple[float, float] | None
    figure_suggestion: str
    interpretation: str


class TfidfRetriever:
    def __init__(self, corpus_texts: List[str]):
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.texts = corpus_texts
        self.matrix = self.vectorizer.fit_transform(corpus_texts) if corpus_texts else None

    def query(self, question: str, k: int = 3) -> List[RetrievalResult]:
        if not self.texts or self.matrix is None:
            return []
        q = self.vectorizer.transform([question])
        sims = cosine_similarity(q, self.matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:k]
        return [RetrievalResult(snippet=self.texts[i], score=float(sims[i])) for i in top_idx]


def load_corpus(corpus_dir: Path) -> List[str]:
    chunks: List[str] = []
    if not corpus_dir.exists():
        return chunks

    for txt_path in sorted(corpus_dir.glob("*.txt")):
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks.extend(paragraphs)
    return chunks


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = math.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(y) - np.mean(x)) / pooled_std)


def mean_diff_ci(x: np.ndarray, y: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    diff = np.mean(y) - np.mean(x)
    sx2 = np.var(x, ddof=1) / len(x)
    sy2 = np.var(y, ddof=1) / len(y)
    se = math.sqrt(sx2 + sy2)
    if se == 0:
        return float(diff), float(diff)
    df = len(x) + len(y) - 2
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    return float(diff - tcrit * se), float(diff + tcrit * se)


def cramers_v(table: np.ndarray) -> float:
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    r, c = table.shape
    denom = n * (min(r - 1, c - 1))
    return float(math.sqrt(chi2 / denom)) if denom > 0 else 0.0


def analyze_dataframe(df: pd.DataFrame) -> AnalysisResult:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

    # Case 1: group comparison (preferred path if available)
    if numeric_cols and categorical_cols:
        outcome = numeric_cols[0]
        group = categorical_cols[0]
        groups = [g.dropna().values for _, g in df.groupby(group)[outcome]]
        labels = [str(name) for name in df[group].dropna().unique()]

        if len(groups) == 2:
            a, b = groups
            shapiro_a = stats.shapiro(a) if len(a) >= 3 else (None, 1.0)
            shapiro_b = stats.shapiro(b) if len(b) >= 3 else (None, 1.0)
            levene = stats.levene(a, b)
            assumptions = [
                f"Shapiro normality {labels[0]} p={shapiro_a.pvalue:.4f}",
                f"Shapiro normality {labels[1]} p={shapiro_b.pvalue:.4f}",
                f"Levene homogeneity p={levene.pvalue:.4f}",
            ]

            normal = shapiro_a.pvalue > 0.05 and shapiro_b.pvalue > 0.05
            equal_var = levene.pvalue > 0.05

            if normal:
                t = stats.ttest_ind(a, b, equal_var=equal_var)
                d = cohens_d(a, b)
                ci = mean_diff_ci(a, b)
                interp = (
                    f"An independent-samples t-test compared {outcome} between {labels[0]} and {labels[1]}. "
                    f"The difference {'was' if t.pvalue < 0.05 else 'was not'} statistically significant "
                    f"(t={t.statistic:.3f}, p={t.pvalue:.4f}, d={d:.3f}, "
                    f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}])."
                )
                return AnalysisResult(
                    test_name="Independent-samples t-test",
                    assumptions=assumptions,
                    statistic_label="t",
                    statistic_value=float(t.statistic),
                    p_value=float(t.pvalue),
                    effect_size_label="Cohen's d",
                    effect_size_value=d,
                    confidence_interval=ci,
                    figure_suggestion="Box plot with jittered individual points by group",
                    interpretation=interp,
                )

            u = stats.mannwhitneyu(a, b, alternative="two-sided")
            n1, n2 = len(a), len(b)
            z = (u.statistic - (n1 * n2 / 2)) / math.sqrt((n1 * n2 * (n1 + n2 + 1)) / 12)
            r = abs(z) / math.sqrt(n1 + n2)
            interp = (
                f"A Mann–Whitney U test compared {outcome} between {labels[0]} and {labels[1]}. "
                f"The difference {'was' if u.pvalue < 0.05 else 'was not'} statistically significant "
                f"(U={u.statistic:.3f}, p={u.pvalue:.4f}, r={r:.3f})."
            )
            return AnalysisResult(
                test_name="Mann–Whitney U",
                assumptions=assumptions,
                statistic_label="U",
                statistic_value=float(u.statistic),
                p_value=float(u.pvalue),
                effect_size_label="r",
                effect_size_value=float(r),
                confidence_interval=None,
                figure_suggestion="Box plot or violin plot by group",
                interpretation=interp,
            )

        if len(groups) > 2:
            normality_ps = [stats.shapiro(g).pvalue if len(g) >= 3 else 1.0 for g in groups]
            normal = all(p > 0.05 for p in normality_ps)
            assumptions = [f"Shapiro p-values by group: {', '.join(f'{p:.4f}' for p in normality_ps)}"]
            if normal:
                f = stats.f_oneway(*groups)
                interp = (
                    f"A one-way ANOVA tested differences in {outcome} across {len(groups)} groups. "
                    f"The omnibus effect {'was' if f.pvalue < 0.05 else 'was not'} significant "
                    f"(F={f.statistic:.3f}, p={f.pvalue:.4f})."
                )
                return AnalysisResult(
                    test_name="One-way ANOVA",
                    assumptions=assumptions,
                    statistic_label="F",
                    statistic_value=float(f.statistic),
                    p_value=float(f.pvalue),
                    effect_size_label="eta^2 (placeholder)",
                    effect_size_value=float("nan"),
                    confidence_interval=None,
                    figure_suggestion="Box plot across all groups",
                    interpretation=interp,
                )

            h = stats.kruskal(*groups)
            interp = (
                f"A Kruskal–Wallis test evaluated group differences in {outcome}. "
                f"The omnibus effect {'was' if h.pvalue < 0.05 else 'was not'} significant "
                f"(H={h.statistic:.3f}, p={h.pvalue:.4f})."
            )
            return AnalysisResult(
                test_name="Kruskal–Wallis",
                assumptions=assumptions,
                statistic_label="H",
                statistic_value=float(h.statistic),
                p_value=float(h.pvalue),
                effect_size_label="epsilon^2 (placeholder)",
                effect_size_value=float("nan"),
                confidence_interval=None,
                figure_suggestion="Box plot across all groups",
                interpretation=interp,
            )

    # Case 2: two numeric columns -> correlation
    if len(numeric_cols) >= 2:
        x = df[numeric_cols[0]].dropna().values
        y = df[numeric_cols[1]].dropna().values
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]

        sx = stats.shapiro(x) if len(x) >= 3 else (None, 1.0)
        sy = stats.shapiro(y) if len(y) >= 3 else (None, 1.0)
        normal = sx.pvalue > 0.05 and sy.pvalue > 0.05
        assumptions = [
            f"Shapiro {numeric_cols[0]} p={sx.pvalue:.4f}",
            f"Shapiro {numeric_cols[1]} p={sy.pvalue:.4f}",
        ]

        if normal:
            r = stats.pearsonr(x, y)
            interp = (
                f"Pearson correlation tested association between {numeric_cols[0]} and {numeric_cols[1]}. "
                f"The correlation {'was' if r.pvalue < 0.05 else 'was not'} significant "
                f"(r={r.statistic:.3f}, p={r.pvalue:.4f})."
            )
            return AnalysisResult(
                test_name="Pearson correlation",
                assumptions=assumptions,
                statistic_label="r",
                statistic_value=float(r.statistic),
                p_value=float(r.pvalue),
                effect_size_label="r",
                effect_size_value=float(abs(r.statistic)),
                confidence_interval=None,
                figure_suggestion="Scatter plot with fitted regression line",
                interpretation=interp,
            )

        s = stats.spearmanr(x, y)
        interp = (
            f"Spearman correlation tested monotonic association between {numeric_cols[0]} and {numeric_cols[1]}. "
            f"The association {'was' if s.pvalue < 0.05 else 'was not'} significant "
            f"(rho={s.statistic:.3f}, p={s.pvalue:.4f})."
        )
        return AnalysisResult(
            test_name="Spearman correlation",
            assumptions=assumptions,
            statistic_label="rho",
            statistic_value=float(s.statistic),
            p_value=float(s.pvalue),
            effect_size_label="|rho|",
            effect_size_value=float(abs(s.statistic)),
            confidence_interval=None,
            figure_suggestion="Scatter plot with rank trend",
            interpretation=interp,
        )

    # Case 3: two categorical columns -> chi-square
    if len(categorical_cols) >= 2:
        c1, c2 = categorical_cols[0], categorical_cols[1]
        table = pd.crosstab(df[c1], df[c2]).values
        chi2, p, _, _ = stats.chi2_contingency(table)
        cv = cramers_v(table)
        interp = (
            f"A chi-square test of independence evaluated association between {c1} and {c2}. "
            f"The association {'was' if p < 0.05 else 'was not'} significant "
            f"(chi2={chi2:.3f}, p={p:.4f}, Cramer's V={cv:.3f})."
        )
        return AnalysisResult(
            test_name="Chi-square test of independence",
            assumptions=["Expected cell counts should generally exceed 5."],
            statistic_label="chi2",
            statistic_value=float(chi2),
            p_value=float(p),
            effect_size_label="Cramer's V",
            effect_size_value=cv,
            confidence_interval=None,
            figure_suggestion="Stacked bar chart or mosaic plot",
            interpretation=interp,
        )

    raise ValueError("Could not infer a supported statistical design from the provided dataframe.")


def build_response(question: str, retrieved: List[RetrievalResult], analysis: AnalysisResult) -> str:
    ctx = "\n".join([f"- ({r.score:.3f}) {r.snippet}" for r in retrieved]) if retrieved else "- No external context retrieved."
    ci_text = (
        f"95% CI: [{analysis.confidence_interval[0]:.3f}, {analysis.confidence_interval[1]:.3f}]"
        if analysis.confidence_interval
        else "95% CI: not available for this test in current implementation"
    )

    assumptions_text = "\n- ".join(analysis.assumptions)

    return f"""
Question: {question}

Retrieved context:
{ctx}

Selected test: {analysis.test_name}
Assumptions checked:
- {assumptions_text}
Statistic: {analysis.statistic_label}={analysis.statistic_value:.4f}
p-value: {analysis.p_value:.6f}
Effect size: {analysis.effect_size_label}={analysis.effect_size_value:.4f}
{ci_text}
Suggested figure: {analysis.figure_suggestion}

Results-style interpretation:
{analysis.interpretation}
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="StatSight RAG prototype")
    parser.add_argument("--data", required=True, help="Path to CSV file")
    parser.add_argument("--question", required=True, help="Research question")
    parser.add_argument("--corpus_dir", default="data/knowledge", help="Directory with .txt knowledge files")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    corpus = load_corpus(Path(args.corpus_dir))
    retriever = TfidfRetriever(corpus)
    retrieved = retriever.query(args.question, k=3)

    analysis = analyze_dataframe(df)
    report = build_response(args.question, retrieved, analysis)
    print(report)


if __name__ == "__main__":
    main()
