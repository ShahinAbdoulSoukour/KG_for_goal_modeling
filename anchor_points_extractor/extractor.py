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

    # extract the goal only once
    goal = goal_triples_df["GOAL"].iloc[0]

    # compute cosine similarity
    goal_embedding = model.encode([goal])
    triple_embeddings = model.encode(goal_triples_df["TRIPLE"].tolist())

    goal_triples_df["SCORE"] = cosine_similarity(
        goal_embedding,
        triple_embeddings
    ).flatten()

    # sort by descending score 
    goal_triples_df = goal_triples_df.sort_values(by="SCORE", ascending=False)

    # extract the highest score
    highest_score = goal_triples_df["SCORE"].iloc[0]

    # filter based on interval [85% – 100% of max]
    score_threshold = highest_score * 0.85
    filtered_df = goal_triples_df[
        goal_triples_df["SCORE"] >= score_threshold
    ]

    # if too few results (< 4), relax threshold to 65% + keep 4 best scores
    if len(filtered_df.index) < 4:
        filtered_df = goal_triples_df[
            goal_triples_df["SCORE"] >= highest_score * 0.65
        ].nlargest(4, "SCORE")

    # return both (sorted) --> highest-scoring anchor points
    return filtered_df, goal_triples_df.reset_index(drop=True)
