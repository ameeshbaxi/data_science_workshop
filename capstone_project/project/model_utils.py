import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


def cv_optimize(clf, parameters, Xtrain, ytrain, n_folds=5):
    """
    Find the best parameter for the classifier using GridSearchCV.
    
    Parameters:
        clf -  Classifier
        parameters - dictionary with classifier parameters
        Xtrain - X training data
        ytrain - y training data
        n_folds - Number of folds (default=5)
    """
    gs = sklearn.model_selection.GridSearchCV(clf, param_grid=parameters, cv=n_folds)
    gs.fit(Xtrain, ytrain)
    print("BEST PARAMS", gs.best_params_)
    best = gs.best_estimator_
    return best

def do_classify(clf, parameters, indf, featurenames, targetname, target1val, standardize=False, train_size=0.8):
    """
    Top Level function to find the best parameters for a
    classifier from a data frame. It uses cv_optimize
    function after creating train / test split.
    
    Parameters:
        clf - Classifier
        parameters - dictionary with classifier parameters
        indf - Data Frame
        featurenames - List of feature names
        targetname - The colum defining the classifier
        target1val - The value of the column (checks if column value > target value)
        standardize - If X data need to standardize (default=False)
        train_size - trainRatio for train / test split (default=0.8)
    """
    subdf=indf[featurenames]
    if standardize:
        subdfstd=(subdf - subdf.mean())/subdf.std()
    else:
        subdfstd=subdf
    X=subdfstd.values
    y=(indf[targetname].values > target1val)*1
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=train_size)
    clf = cv_optimize(clf, parameters, Xtrain, ytrain)
    clf = clf.fit(Xtrain, ytrain)
    training_accuracy = clf.score(Xtrain, ytrain)
    test_accuracy = clf.score(Xtest, ytest)
    print("Accuracy on training data: {:0.2f}".format(training_accuracy))
    print("Accuracy on test data:     {:0.2f}".format(test_accuracy))
    return clf, Xtrain, ytrain, Xtest, ytest

