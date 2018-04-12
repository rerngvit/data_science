import pandas as pd


def create_feature_outcome_pd(column_names, scores,
                              feature_name="Feature",
                              score_name="Score"):
    """ Create a Pandas dataframe summarizing feature important scores.

    :param column_names: list of column names
    :param scores: list of scores (with the corresponding column_names)
    :param feature_name: the column name for features of the output data frame
    :param score_name: the column name for scores of the output data frame
    :return:
    """
    impact_outcome_pd = pd.DataFrame([column_names, scores],
                                 index=[feature_name, score_name]
                                 ).transpose().sort_values(
                                 by=score_name, ascending=False)
    return impact_outcome_pd
