import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from . import feature_importance


def _single_iteration_pred_recall(selected_model, data_retrieval_func):
    """ A single iteration of computing precision and recall for a selected model

    :param selected_model: a SKLearn model that has a predict_proba function
    :param data_retrieval_func: a function returning an output of
        X_train, Y_train, X_test, Y_test
    """
    X_train, Y_train, X_test, Y_test = data_retrieval_func()
    """ Compute average precision recall"""
    selected_model.fit(X_train, Y_train)
    Y_predict_prob = selected_model.predict_proba(X_test)[:, 1]

    precisions, recalls, _ = precision_recall_curve(Y_test, Y_predict_prob)

    data_se = pd.Series(precisions, index=recalls)
    aucpr = average_precision_score(Y_test, Y_predict_prob)

    if hasattr(selected_model, "feature_importances_"):
        feature_column_pd = feature_importance.create_feature_outcome_pd(X_train.columns, selected_model.feature_importances_)
    else:
        feature_column_pd = None

    return data_se, aucpr, Y_test, feature_column_pd


def multiple_iterations_pred_recall(selected_model, data_retrieval_func,
                                    num_iterations=3):
    """ Executing multiple computations of precision and recall for a selected model

    :param selected_model: a selected classification model (SKlearn estimator)
    :param data_retrieval_func: a data retrieval function returning an output of
        X_train, Y_train, X_test, Y_test
    :param num_iterations: the number of iterations to repeatedly compute precision and recall
    :return:
    """
    list_of_data_se = []
    list_of_Y_test  = []
    list_of_feature_column_pd = []
    for _ in range(num_iterations):
        data_se, aucpr, Y_test, feature_column_pd = _single_iteration_pred_recall(
            selected_model, data_retrieval_func=data_retrieval_func)
        list_of_data_se.append(data_se)
        list_of_Y_test.append(Y_test)
        list_of_feature_column_pd.append(feature_column_pd)
    return list_of_data_se, list_of_Y_test, list_of_feature_column_pd


def compute_aggregation_data(list_of_data_se):
    """Compute aggregation dictionary putting recall_value as key

    :param list_of_data_se: list of raw_data for aggregation
    :return a summary dictionary for all data
    """
    offset_consider = 0.005
    results_dict = {}
    for recall_i in list_of_data_se[0].index.unique():
        for j in range(len(list_of_data_se)):
            matched_index = np.apply_along_axis(
                lambda x: (x >= recall_i - offset_consider) &
                          (x <= recall_i + offset_consider), 0,
                list_of_data_se[j].index)

            if matched_index.sum() > 0:
                precision_j_at_recall_i = list_of_data_se[j].loc[matched_index].mean()
                print("[%s] Found precision %s at recall %s" % (
                    j, precision_j_at_recall_i, recall_i))

                if recall_i not in results_dict.keys():
                    results_dict[recall_i] = [precision_j_at_recall_i]
                else:
                    assert (type(results_dict[recall_i]) == list)
                    results_dict[recall_i].append(precision_j_at_recall_i)
            else:
                print("[%s] Do not find precision at recall %s" % (j, recall_i))
    return results_dict
