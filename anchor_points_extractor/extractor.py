"""
Anchor points extractor
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def anchor_points_extractor(
    goal_triples_df: pd.DataFrame,
    model: SentenceTransformer,
    filtered_out_triples: list[str],
) -> pd.DataFrame:
    """
    Calculating the cosine similarity between the triple and the formulated goal.

    Parameters
    ----------
    goal_triples_df :           pd.DataFrame
                                DataFrame containing triples and the formulated goal
    model :                     SentenceTransformer
                                Transformer model
    filtered_out_triples :      list[str]
                                list of selected triples by the user (history of triples)

    Returns
    -------
    pd.DataFramegit stat
    DataFrame containing the similarity scores.
    """
    goal_triples_df["SCORE"] = goal_triples_df.apply(
        lambda row: pd.Series(
            cosine_similarity(
                model.encode(row["GOAL"]).reshape(1, -1),  # goal embedding
                model.encode(row["TRIPLE"]).reshape(1, -1),  # simple triples embedding
            ).flatten()
            if not filtered_out_triples or row["TRIPLE"] not in filtered_out_triples
            else 0
        ),
        axis=1,
    )

    highest_score = goal_triples_df["SCORE"].max()

    # set an interval [highest_score * 0.85, highest_score]
    score_interval = [highest_score * 0.85, highest_score]

    # filter the dataframe based on the score interval
    filtered_df = goal_triples_df[
        (goal_triples_df["SCORE"] >= score_interval[0])
        & (goal_triples_df["SCORE"] <= score_interval[1])
    ]

    if len(filtered_df.index) < 4:
        filtered_df = goal_triples_df[
            goal_triples_df["SCORE"] >= highest_score * 0.65
        ].nlargest(4, "SCORE")

    return filtered_df, goal_triples_df
