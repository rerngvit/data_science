import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr, pearsonr
import pandas as pd


def visualize_missing_value(df):
    """ Visualize amount of missing values with a bar chart

    :param df: an input Pandas data frame to investigate missing values
    """
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20) 
    
    plt.figure(figsize=(20,20))
    missing_fraction_se = df.apply(lambda x: (x.isnull().sum() + 0.0) / x.shape[0], 
                                    axis=0)
    missing_fraction_se[missing_fraction_se >= 0.05
                       ].sort_values(ascending=True).plot(kind="barh")
    plt.show()


def compute_correlation_with_missing_values(data_pd, feature_with_missing_values, other_feature):
    """Compute correlation score according to the feature_with_missing_values and another feature

    :param feature_with_missing_values: the source feature to compare with
    :param other_feature: another feature to consider
    
    Return:
        Correlation score between 0, 1
    """
    is_feature_missing = data_pd[feature_with_missing_values].isnull().apply(lambda x: 1 if x else 0)
    other_feature_se = data_pd[other_feature]
    other_feature_existed_idx = ~other_feature_se.isnull()

    other_feature_existed_se = other_feature_se.loc[other_feature_existed_idx]
    is_missing_existed_se = is_feature_missing.loc[other_feature_existed_idx]
    if not (is_missing_existed_se.nunique() == 1 or 
            other_feature_existed_se.nunique() ==1):
        cor_result = kendalltau(is_missing_existed_se, other_feature_existed_se, nan_policy="raise")[0]
    else:
        print("is_missing_existed_se or other_feature_existed_se is constant!!!")
        cor_result = 0
    return abs(cor_result)


def compute_correlation_heatmap_with_missing_values_pd(
        input_pd, features_with_missing_values, all_features):
    """ Compute correlation heatmap with missing values input data frame

    :param input_pd: an input Pandas data frame
    :param features_with_missing_values: list of feature names with missing values
    :param all_features: list of all features
    :return: a Pandas data frame indicating correlations with the feature with missing values
    """

    list_of_list_of_scores = []
    print("features_with_missing_values has size ", len(features_with_missing_values))
    for feature_with_missing in features_with_missing_values:
        list_of_scores = []
        for feature in all_features:
            cor_score = compute_correlation_with_missing_values(input_pd, feature_with_missing, feature)
            list_of_scores.append(cor_score)
        list_of_list_of_scores.append(list_of_scores)
    
    heatmap_pd = pd.DataFrame(list_of_list_of_scores,
                              columns=all_features, 
                              index=features_with_missing_values)
    return heatmap_pd

