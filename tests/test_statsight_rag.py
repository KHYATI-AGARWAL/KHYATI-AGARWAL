import pandas as pd

from src.statsight_rag import analyze_dataframe


def test_two_group_analysis_selects_ttest_or_mannwhitney():
    df = pd.DataFrame(
        {
            "group": ["control"] * 6 + ["treatment"] * 6,
            "outcome": [10.0, 9.9, 10.1, 10.2, 10.0, 9.8, 11.0, 10.9, 11.2, 11.1, 10.8, 11.0],
        }
    )
    result = analyze_dataframe(df)
    assert result.test_name in {"Independent-samples t-test", "Mann–Whitney U"}
    assert isinstance(result.p_value, float)


def test_categorical_association_uses_chisquare():
    df = pd.DataFrame(
        {
            "arm": ["A", "A", "B", "B", "B", "A", "A", "B"],
            "response": ["yes", "no", "yes", "yes", "no", "no", "yes", "no"],
        }
    )
    result = analyze_dataframe(df)
    assert result.test_name == "Chi-square test of independence"
