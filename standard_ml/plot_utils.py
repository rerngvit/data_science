import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def visualize_precision_recall(y_predict_prob, y_actual, show_base_line=True,
                               fill_style="area_below", precision_offset=0.05,
                               font_size=20):
    """ Visualize precision recall curve with base line (flipping a coin)

    :param y_predict_prob: numpy array of predicted probabilities
    :param y_actual: numpy array of ground-truth outcomes
    :param show_base_line: a boolean whether to visualize the base line (flipping a coin)
    :param fill_style: either "area_below" or "uncertainty_offset"
    :param precision_offset: constant precision offset
        (only applicable for fill_style = "uncertainty_offset")
    :param font_size: font size for visualization

    """
    precision, recall, thresholds = precision_recall_curve(y_actual, y_predict_prob)
    average_precision = average_precision_score(y_actual, y_predict_prob)
    
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    plt.rc('legend', fontsize=font_size)
    plt.rc('axes', labelsize=font_size)
    plt.step(recall, precision, color='b', alpha=0.2),

    if fill_style == "area_below":
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    elif fill_style == "uncertainty_offset":
        plt.fill_between(recall, precision - precision_offset,
                         precision + precision_offset, alpha=0.1,
                         color="b")

    if show_base_line:
        plt.axhline(y=average_precision_score(y_actual, np.repeat(0.5, y_actual.shape[0])), color='r', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))


def visualize_precision_recall_with_variations(list_of_y_test, results_dict,
                                               show_base_line=True,
                                               font_size=50):
    """  Visualize precision recall curve with variation scores (from random samplings)

    :param list_of_y_test: List of ground truth outcomes
    :param results_dict: Dictionary of results having recall value as key
    :param show_base_line: a flag whether to visualize a base line or not
    :param font_size: font size for visualization
    """
    plt.figure(figsize=(20, 20))
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    plt.rc('legend', fontsize=font_size)
    plt.rc('axes', labelsize=font_size)

    recalls = np.array(sorted(results_dict.keys()))
    precision_means = np.array([np.array(results_dict[recall_i]).mean()
                                for recall_i in recalls.tolist()])
    precision_stds = np.array([np.array(results_dict[recall_i]).std()
                               for recall_i in recalls.tolist()])
    plt.plot(recalls, precision_means, color='b')
    plt.fill_between(recalls, precision_means - precision_stds,
                     precision_means + precision_stds, alpha=0.2,
                     color="b")

    if show_base_line:
        flipping_coin_precisions = np.array([average_precision_score(
            Y_actual, np.repeat(0.5, Y_actual.shape[0])) for Y_actual in list_of_y_test])

        plt.axhline(y=flipping_coin_precisions.mean(), color='r', linestyle='--')
        plt.fill_between(recalls, flipping_coin_precisions.mean() - flipping_coin_precisions.std(),
                         flipping_coin_precisions.mean() + flipping_coin_precisions.std(), alpha=0.2,
                         color="r")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    avg_weighted_precision = (np.diff(recalls, n=1) * precision_means[:-1]).sum()
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
        avg_weighted_precision), fontsize=30)


def visualize_ROC_curve(Y_predict_prob, Y_actual, font_size=40):
    """ Utilization code for visualizing ROC Curve

    :param Y_predict_prob: numpy array of predicted probabilities
    :param Y_actual: numpy array of ground truth outcomes
    :param font_size: font_size for visualization
    """
    
    fpr, tpr, _ = metrics.roc_curve(Y_actual, Y_predict_prob)
    roc_auc  =    metrics.roc_auc_score(Y_actual, Y_predict_prob)
    plt.figure(figsize=(20,15))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC curve)', fontsize=font_size)
    plt.legend(loc="lower right")
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    plt.rc('legend', fontsize=font_size)
    plt.rc('axes', labelsize=font_size)
    plt.show()


def visualize_feature_outcome_pd(impact_outcome_pd,
                                 feature_name="Feature",
                                 score_name="Score",
                                 font_size=50):
    """ Visualize feature importance pandas dataframe as a bar chart

    :param impact_outcome_pd: feature importance data frame (from feature_importance)
    :param feature_name: column name for feature name
    :param score_name: column name for feature importance score
    :param font_size: font_size for visualization
    """

    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    fig = plt.figure(figsize=(20, 20))
    ax = fig.gca()
    impact_outcome_pd.sort_values(by=score_name, ascending=False).iloc[:10, :].plot(
        x=feature_name, y=score_name, kind="barh", ax=ax, legend=False)
    plt.title("Feature importances")
    plt.xlabel('Score', fontsize=font_size)
    plt.ylabel('Feature', fontsize=font_size)
    plt.show()

# Credit: adapted from SKLearn example website
# Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5),
                        scorer=make_scorer(f1_score), scorer_caption="F1 Score"):
    """
    Generate a simple plot of the test and training learning curve.


    :param estimator: object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    :param title : string
    :param X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    :param y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    :param ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    :param cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    :param train_sizes: numpy array indicating fraction of rows to evaluate
    :param n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    :param scorer : sklearn scorer to use for visualization
    :param scorer_caption : caption to describe the scorer
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel(scorer_caption)

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scorer, n_jobs=n_jobs,  train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt