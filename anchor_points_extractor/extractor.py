"""
Anchor points extractor from the Knowledge Graph
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
    Calculating the cosine similarity between the triples and the formulated goal by the designer.

    Parameters
    ----------
    goal_triples_df :           pd.DataFrame
                                DataFrame containing triples and the formulated goal
    model :                     SentenceTransformer
                                Transformer model
    filtered_out_triples :      list[str]
                                list of filtered triples for the subgoal creation (history of selected triples)

    Returns
    -------
    Two DataFrames containing the anchor points, the formulated goal and the similarity scores.
    filtered_df:                pd.DataFrame
                                DataFrame containing the most relevant anchor points - the top candidates triples
    goal_triples_df:            pd.DataFrame
                                DataFrame containing all anchor points
    """

    # remove filtered (or selected) triples from the goal_triples_df DataFrame entirely
    goal_triples_df = goal_triples_df[~goal_triples_df["TRIPLE"].isin(filtered_out_triples)].copy()

    goal = goal_triples_df["GOAL"].iloc[0]

    goal_triples_df["SCORE"] = pd.Series(cosine_similarity(
                model.encode([goal]),  # goal embedding
                model.encode(goal_triples_df["TRIPLE"].to_list()),  # triples (as str) embedding
            ).flatten(),
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
