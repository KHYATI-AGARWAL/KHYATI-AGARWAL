# StatSight RAG (Statistical Results Interpreter + Figure Checker)

This repository provides a practical starter implementation for your **StatSight** idea: a retrieval-augmented assistant that combines:

1. **Evidence retrieval (RAG)** over method/reporting guidelines.
2. **Programmatic statistical analysis** on tabular data (CSV).
3. **Results-section style writing** with assumptions, effect size, confidence intervals, and plot suggestions.

## Why this design works for a research paper

- It avoids "hallucinated stats" by computing every statistic directly from data.
- It keeps responses grounded by retrieving relevant methodological snippets.
- It is easy to reproduce (single CLI command, deterministic analysis code).

## Project structure

- `src/statsight_rag.py` — main pipeline + CLI.
- `data/knowledge/` — put your paper/reporting guidelines here (`.txt` files).
- `data/sample_experiment.csv` — tiny demo dataset.
- `requirements.txt` — dependencies.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/statsight_rag.py \
  --data data/sample_experiment.csv \
  --question "Is treatment better than control?" \
  --corpus_dir data/knowledge
```

## Input format suggestions

The analyzer auto-detects roles from columns, but best practice is to provide one of these layouts:

1. **Two-group comparison**
   - `group` (categorical, e.g., control/treatment)
   - `outcome` (numeric)
2. **Multi-group comparison**
   - `group` + `outcome`
3. **Association test**
   - two categorical columns
4. **Correlation**
   - two numeric columns

## What the tool returns

- Selected test and rationale.
- Assumption checks (Shapiro normality, Levene variance homogeneity when relevant).
- Test statistic, p-value, confidence interval (where implemented), effect size.
- Suggested figure type.
- Publication-ready text block for a Results section.
- Retrieved context snippets used for grounding.

## Research-paper framing ideas

You can report this as a **hybrid neuro-symbolic analysis assistant**:

- Retrieval component: TF-IDF similarity retrieval over domain/reporting knowledge.
- Symbolic/statistical component: explicit test selection and numerical computation with SciPy/StatsModels.
- Natural-language component: template-based scientific reporting constrained by computed values.

Potential evaluation axes:

- Test selection accuracy vs. statisticians.
- Numerical correctness (exact match to known outputs).
- Reporting completeness (includes effect size + assumptions).
- Hallucination rate vs. generic LLM baseline.

## Limitations and next steps

- Current retriever is TF-IDF (swap with embedding model/vector DB for stronger semantic search).
- Add multiple-comparison corrections and mixed-effects models.
- Add figure generation and automated figure-check consistency tests.
- Add a grading harness against benchmark datasets.
