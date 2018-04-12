# Executing random search on parameters
from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn import metrics
f1_scorer = metrics.make_scorer(metrics.f1_score)
roc_auc_scorer = metrics.make_scorer(metrics.roc_auc_score)
precisions_scorer = metrics.make_scorer(metrics.precision_score)



def report_search_scores(results, n_top=5):
    """ Report summary of search scores
    # Credit: source code adapted from SKLearn
    # Adapt from http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py

    :param results: CV results from RandomizedSearchCV
    :param n_top: number of scores to report from the top
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def auto_tune_paras_random_search(model, param_dist, x_input_train, y_input_train, n_iter_search=1, num_folds=5):
    """ Executing random search of the input model according to the param dictionary
    # Credit: source code adapted from SKLearn
    # Adapt from http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py

    :param model: a sklearn model (an Estimator)
    :param param_dist: parameter dictionary
    :param x_input_train: A pandas data frame of input features for the train set
    :param y_input_train: A numpy array or pandas series of ground truth for the train set
    :param n_iter_search: number of iterations to search
    :param num_folds: number of folds to do cross validation
    :return: trained model from the cross validation
    """

    random_search_pipe = RandomizedSearchCV(model, param_distributions=param_dist, scoring=f1_scorer, n_iter=n_iter_search, verbose=10, cv=num_folds, random_state=0)
    start = time()
    random_search_pipe.fit(x_input_train, y_input_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report_search_scores(random_search_pipe.cv_results_)
    return (random_search_pipe)
