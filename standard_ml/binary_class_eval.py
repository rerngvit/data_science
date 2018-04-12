import numpy as np
from sklearn import metrics
from . import feature_importance
from . import plot_utils

def report_classification_results(Y_actual, Y_prediction):
    """ Report over all binary classification results (including confusion matrix)

    :param Y_actual: A numpy array or pandas series of ground truth outcomes
    :param Y_prediction: A numpy array or pandas series of predictions
    """
    print("accuracy = %.2f, precision = %.2f, recall = %.2f, F1-score = %.2f " % (
            metrics.accuracy_score(Y_actual, Y_prediction),
            metrics.precision_score(Y_actual, Y_prediction),
            metrics.recall_score(Y_actual, Y_prediction),
            metrics.f1_score(Y_actual, Y_prediction)
        ))
    print(" ========= Confusion Matrix ================")
    print(metrics.confusion_matrix(Y_actual, Y_prediction))
    print(" ===========================================")


def compute_classification_metrics(Y_actual, Y_prediction):
    """ Compute basic binary classification metrics.

    :param Y_actual: A numpy array or pandas series of ground truth outcomes
    :param Y_prediction: A numpy array or pandas series of predictions
    :return: a dictionary with key of accuracy, precision, recall, and f1_score
    """

    print(metrics.confusion_matrix(Y_actual, Y_prediction))
    return {
        "accuracy":  metrics.accuracy_score(Y_actual, Y_prediction),
        "precision": metrics.precision_score(Y_actual, Y_prediction),
        "recall": metrics.recall_score(Y_actual, Y_prediction),
        "f1_score":  metrics.f1_score(Y_actual, Y_prediction),
    }

def naive_sampling_results(Y_class_input):
    """ Compute naive sampling result (flipping a coin)

    :param Y_class_input: A numpy array or pandas series of ground truth outcomes
    :return:
    """
    return compute_classification_metrics(Y_class_input,
                                          np.random.randint(0, high=2, size=Y_class_input.shape))


def eval_trained_model(trained_model, X_input_test, Y_input_test):
    """ Evaluate trained model on specified input and output

    :param trained_model: a trained sklearn model (an Estimator)
    :param X_input_test: A pandas data frame of input features for the test set
    :param Y_input_test: A numpy array or pandas series of ground truth for the test set
    :return:
    """

    print(" ======== Classifier Test results ====================")
    test_predictions = trained_model.predict(X_input_test)
    report_classification_results(Y_input_test, test_predictions)
    print(" ==========================================")

    Y_predict_prob = trained_model.predict_proba(X_input_test)[:, 1]

    plot_utils.visualize_ROC_curve(Y_predict_prob, Y_input_test)

    plot_utils.visualize_precision_recall(Y_predict_prob, Y_input_test)

    return test_predictions


def binary_classify_train_test(model, X_input_train, Y_input_train,
                               X_input_test, Y_input_test):
    """ Train a SKlearn classifier on the train set and evaluate on the test set.

    :param model: a sklearn model (an Estimator)
    :param X_input_train: A pandas data frame of input features for the train set
    :param Y_input_train: A numpy array or pandas series of ground truth for the train set
    :param X_input_test: A pandas data frame of input features for the test set
    :param Y_input_test: A numpy array or pandas series of ground truth for the test set
    :return: (feature_importance_pd, train_predictions, test_predictions)
    """

    model.fit(X_input_train, Y_input_train)
    
    print(" ======== Classifier Train results ===================")
    print(" ==========================================")
    
    train_predictions = model.predict(X_input_train)
    report_classification_results(Y_input_train, train_predictions)

    test_predictions = eval_trained_model(trained_model=model,
                       X_input_test=X_input_test,
                       Y_input_test=Y_input_test)

    feature_column_pd = None
    if hasattr(model, "feature_importances_"):
        feature_column_pd = feature_importance.create_feature_outcome_pd(X_input_train.columns, model.feature_importances_)
        plot_utils.visualize_feature_outcome_pd(feature_column_pd)
    
    return feature_column_pd, train_predictions, test_predictions


def anomaly_predict_converter(predict_np):
    """ Utility class to convert predictions to convert predictions to anomaly detections

    :param predict_np: prediction numpy array
    :return: a numpy array of anomaly prediction
    """
    def sklearn_anomaly_convert_rule(y):
        if y == 1:
            return 0
        else:
            return 1
    return np.array([sklearn_anomaly_convert_rule(y) for y in predict_np.tolist()])


def anomaly_train_test(model, X_input_train, Y_input_train,
                               X_input_test, Y_input_test):
    """ Train a SKlearn anomaly detector on the train set and evaluate on the test set.

    :param model: a SKlearn anomaly detector model
    :param X_input_train: A pandas data frame of input features for the train set
    :param Y_input_train: A numpy array or pandas series of ground truth for the train set
    :param X_input_test: A pandas data frame of input features for the test set
    :param Y_input_test: A numpy array or pandas series of ground truth for the test set
    :return: (train_predictions, test_predictions)
    """
    model.fit(X_input_train, Y_input_train)

    print(" ======== Anomaly Train results ===================")
    print(" ==========================================")

    train_predictions = anomaly_predict_converter(model.predict(X_input_train))
    report_classification_results(Y_input_train, train_predictions)

    print(" ======== Anomaly Test results ====================")
    test_predictions = anomaly_predict_converter(model.predict(X_input_test))
    report_classification_results(Y_input_test, test_predictions)
    print(" ==========================================")

    return train_predictions, test_predictions