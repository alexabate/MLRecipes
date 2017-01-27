"""Useful functions for machine learning, some stolen from
   stockexchange/stockexchange/train.py


"""
import sklearn
SKL_VERSION = sklearn.__version__
SKL_VER_NUM = float(SKL_VERSION.split('.')[0] + '.' + SKL_VERSION.split('.')[1])

from sklearn.metrics import (roc_auc_score, accuracy_score,
	                         precision_recall_fscore_support)
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MaxAbsScaler
if SKL_VER_NUM < 0.18:
    from sklearn import cross_validation as ms
    from sklearn.grid_search import GridSearchCV
else:
    from sklearn import model_selection as ms
    from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd


def make_onehot_features(df, feature_columns, features=[]):
    """Create one hot features for each sample in the dataframe by
       creating a list of feature: value mappings for each sample.

       Each time a "new value" for a feature is seen an entire new feature
       is created and for that sample its value is set to 1.

       Then features are ready for input into the DictVectorizer, and it will
       implicitly make these group of features a "one hot" features.

       E.g. if a particular feature is say "gender" and the possible values
       are "female", "male", "non-binary", "other" each sample in the data set
       will now contain four features such that for someone with gender=female:

       "feature_female" = 1
       "feature_male" = 0
       "feature_nonbinary" = 0
       "feature_other" = 0


       Parameters
       ----------
       df : pandas dataframe
            dataframe containing the samples
       feature_columns : list of strings
                         column names to make categorical features from
       features : list of dicts
                  (optional) existing features for each sample

       Returns
       --------
       features : list of dicts
                  dictionary of feature: value for each sample
                  (note that not every sample has every feature in its dict)
    """

    # if features list is un-initialised
    icol = 0
    if len(features)==0:
        features = [{val: 1} for val in df[feature_columns[icol]].values]
        icol = 1

    # for each sample, the VALUE of the feature in the feature column
    # becomes the 'feature name'. The feature value is always 1
    # this is a way of 'one-hot' encoding data that are categorical.
    for i in range(icol, len(feature_columns)):

        for row, val in zip(features, df[feature_columns[i]].values):
            row.update(**{val: 1})

    return features


def make_normal_features(df, feature_columns, features=[]):
    """Create features for each sample in the dataframe by
       creating a list of feature: value mappings for each sample.

       Then features are ready for input into the DictVectorizer

       Parameters
       ----------
       df : pandas dataframe
            dataframe containing the samples
       feature_columns : list of str
                         column names to make categorical features from
       features : list of dicts
                  (optional) existing features for each sample

        Returns
        --------
        features : list of dicts
                   dictionary of feature: value for each sample
                   (note that not every sample has every feature in its dict)
    """

    # if features list is un-initialised
    icol = 0
    if len(features)==0:
        features = [{feature_columns[icol]: val} for val in df[feature_columns[icol]].values]
        icol = 1

    # for each sample, the 'feature name' is the column name and the value
    # is the sample's value in that column
    for i in range(icol, len(feature_columns)):

        for row, val in zip(features, df[feature_columns[i]].values):
            row.update(**{feature_columns[i]: val})

    return features


def create_data_for_model(df, features, column_to_predict):
    """Prepare data to be used in the learning model by transforming the lists
       of feature-value dictionaries (i.e. those created in
       make_normal_features and make_onehot_features) to vectors.

       When feature values are strings, the DictVectorizer will do a binary
       one-hot (aka one-of-K) coding: one boolean-valued feature is constructed
       for each of the possible string values that the feature can take on.

       Parameters
       ----------
       df : pandas dataframe
            DataFrame containing the samples (inputs and outputs)
       features : a list of dicts
                  corresponding to features wanted from each row of the df.
                  Each dictionary is a 'feature': value made from every sample in
                  the df
       column_to_predict : str
                           name of output column to predict

       Returns
       --------
       X : sparse matrix
           the inputs (vectorised)
       y : numpy array
           the outputs
       dv : DictVectorizer
            holds mappings of the feature names to the feature indices in the
            matrix
       mabs : MaxAbsScaler
              how each features was scaled
    """

    dv = DictVectorizer()
    X = dv.fit_transform(features)
    mabs = MaxAbsScaler() # scale each feature by its max absolute value
    X = mabs.fit_transform(X)
    y = df[column_to_predict].astype(np.int64).values

    return X, y, dv, mabs


def gridded_cv_fit(X_train, y_train, grid_search_params, learner, nfolds=10):
    """Perform a cross-valiation grid search of hyperparameters via
       StratifiedKFold

       Parameters
       ----------
       X_train : array of some kind
                 training sample inputs
       y_train : array of some kind
                 training sample known outputs
       grid_search_params : dict
                            dictionary with the learner's
                            hyper_parameter_name's as keys, with the list of
                            values to try as the values
       learner : object
                 the learning algorithm
       nfolds : int
                number of folds
    """

    if SKL_VER_NUM < 0.18:
        cv = ms.StratifiedKFold(y_train, nfolds)
        gs = GridSearchCV(learner, grid_search_params,
                          scoring='roc_auc', cv=cv,
                          n_jobs=-1)
    else:
        cv = ms.StratifiedKFold(nfolds)
        gs = GridSearchCV(learner, grid_search_params,
                          scoring='roc_auc', cv=cv,
                          n_jobs=-1)

    gs.fit(X_train, y_train)
    return gs


def fraction_correct(yp, yt):
    """Return fraction of predictions that are the same as the true values
       (value to predict is 1D)
    """
    return len(yp[yp==yt])/float(len(yp))


def classification_accuracy(y_truth, y_predict):
    """Return the percentage of samples correctly classified. If binary
       classifier return the percentage of true class 0 correctly classified
       and percentage of true class 1 correctly classified

       Parameters
       ----------
       y_truth : array (1-d)
                 the true class values
       y_predict : array (1-d)
                   the predicted class values

       Returns
       --------
       frac_correct : float
                      fraction of samples classed correctly
       frac_correct_class0 : float
                             (if binary) fraction of samples with true class 0
                             classed correctly
       frac_correct_class1 : float
                             (if binary) fraction of samples with true class 1
                             classed correctly
    """
    frac_correct = fraction_correct(y_predict, y_truth)
    msg = "Percentage correct predictions = {:.2f}".format(100.*frac_correct)
    print(msg)

    frac_correct_class0 = np.nan
    frac_correct_class1 = np.nan
    # check if binary classifier
    if len(np.unique(y_truth))==2:

        i_class_0 = y_truth == np.unique(y_truth)[0]
        yp = y_predict[i_class_0]
        yt = y_truth[i_class_0]
        frac_correct_class0 = fraction_correct(yp, yt)

        i_class_1 = y_truth == np.unique(y_truth)[1]
        yp = y_predict[i_class_1]
        yt = y_truth[i_class_1]
        frac_correct_class1 = fraction_correct(yp, yt)

        msg = "Percentage correct predictions (true class 0) = {:.2f}"
        print(msg.format(100.*frac_correct_class0))
        msg = "Percentage correct predictions (true class 1) = {:.2f}"
        print(msg.format(100.*frac_correct_class1))

    return frac_correct, frac_correct_class0, frac_correct_class1


def print_cross_validation_metrics(X, y, learner):
    """Print results of cross-validation to the screen

       Parameters
       ----------
       X : dict
           containing key 'X_test' and optionally also key 'X_train'
           that hold the test and training sample inputs respectively
       y : dict
           containing key 'y_test' and optionally also key 'y_train'
           that hold the test and training sample outputs respectively
       learner : object
                 the learning algorithm

       Returns
       -------
       metrics : list of dict
                 value of metrics at each threshold
    """

    try:
        # Overall best AUC from the learner
        print('Best cross-validation roc_auc: {:4.5f}'.format(learner.best_score_))
    except:
        print('No cross-validation done on learner')

    # Check if training sample present
    if ('X_train' in X) and ('y_train' in y):

        X_train = X['X_train']
        y_train = y['y_train']

        # AUC on training sample
        y_preds = learner.predict_proba(X_train)[:, 1]
        print('Train roc_auc: {:4.5f}'.format(roc_auc_score(y_train, y_preds)))

    X_test = X['X_test']
    y_test = y['y_test']

    # AUC on testing sample
    y_preds = learner.predict_proba(X_test)[:, 1]
    print('Test roc_auc: {:4.5f}'.format(roc_auc_score(y_test, y_preds)))

    print('Summary statistics')
    metrics = []
    thresholds = np.linspace(0.1, 0.9, 9)
    for t in thresholds:
        (prec, rec,
         f1, sup) = precision_recall_fscore_support(y_test,
                                                    y_preds > t,
                                                    average='binary')
        acc = accuracy_score(y_test, y_preds > t)
        metrics.append({'threshold': t,
                        'accuracy': acc,
                        'precision': prec,
                        'recall': rec,
                        'f1_score': f1})

    print(pd.DataFrame(metrics).set_index('threshold').sort_index())

    return metrics


def plot_metrics(metrics, ax):
    """Plot the classiciation metrics as a function of classifier threshold

       Parameters
       ----------
       metrics : list of dict
                 value of each metric at each threshold
       ax : matplotlib axes handle
            axes handle to plot on
    """

    thresholds = [t['threshold'] for t in metrics]
    accuracy = [t['accuracy'] for t in metrics]
    precision = [t['precision'] for t in metrics]
    recall = [t['recall'] for t in metrics]
    f1_score = [t['f1_score'] for t in metrics]


    ax.plot(thresholds, accuracy, color='red',
            label='accuracy')
    ax.plot(thresholds, precision, color='blue', linestyle='dashed',
            label='precision')
    ax.plot(thresholds, recall, color='blue', linestyle='dotted',
            label='recall')
    ax.plot(thresholds, f1_score, color='black',
            label='F1')
    ax.legend(loc='lower left')


def print_features_importances(learner, dv, max_print=50):
    """Print the importance of each features in the DictVectorizer in order
       of most to least important

       Parameters
       ----------
       rf : sklearn.RandomForestClassifier or sklearn.LogisticRegression
            random forest or logistic regression classifier
       dv : DictVectorizer
            holds the features
       max_print : int
            maximum number of features to print
    """

    try:
        feature_importances = learner.feature_importances_
    except:
        feature_importances = learner.coef_[0]

    for icnt, i in enumerate(np.argsort(-1*feature_importances)):
        print(dv.feature_names_[i] , feature_importances[i])

        if icnt==max_print:
            break


def choose_best_threshold(metrics, metric_to_optimise='accuracy'):
    """Using output from `print_cross_validation_metrics`, select optimal
       classifier threshold based on highest score of desired metrics

       Parameters
       -----------
       metrics : list of dict
                 value of each metric at each threshold
       metric_to_optimise : str
                            name of metric to optimise
    """
    thresholds = [t['threshold'] for t in metrics]
    metric = [t[metric_to_optimise] for t in metrics]

    imax = np.argmax(metric)
    return thresholds[imax]
