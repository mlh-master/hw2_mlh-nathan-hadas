import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib as path
import pickle
import sys
import matplotlib as mpl

from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, hinge_loss
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

def check_penalty(penalty='none'):
    """"
    :param penalty: the type of penalty we will use- 'l1', 'l2', 'none'. default is none
    :return solver - the solver type for the Logistic-Regression
    -- taken from tutorial 5, written by Moran Davoodi with the assitance of Yuval Ben Sason & Kevin Kotzen -- 
    """""
    if penalty == 'l1':
        solver = 'liblinear'
    if penalty == 'l2' or penalty == 'none':
        solver = 'lbfgs'
    return solver


def logReg_optimization(penalty, X_train, Y_train, x_test, y_test):
    """"
    :param penalty: the type of penalty we will use- 'l1', 'l2' or 'none'.
    :param X_train: the training dataset.
    :param Y_train: the training labels.
    :param x_test:  the testing dataset.
    :param y_test:  the testing labels. 
    :return chosen_clf - the best model.
    :return clf        - all the testing models.

    """""
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, random_state=10, shuffle=True)
    max_iter = 2000
    solver = check_penalty(penalty)
    log_reg = LogisticRegression(random_state=5, max_iter=max_iter, solver=solver)
    lmbda = np.array([0.01, 1, 10, 100])

    # Todo - ask if need to scale - ('scale', StandardScaler()),
    pipe = Pipeline(steps=[('logistic', log_reg)])
    clf = GridSearchCV(estimator=pipe, param_grid={'logistic__C': 1 / lmbda, 'logistic__penalty': [penalty]},
                       scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'], cv=skf,
                       refit='roc_auc', verbose=0, return_train_score=True)

    clf.fit(X_train, Y_train.squeeze())
    best_model = clf.best_params_

    chosen_clf = clf.best_estimator_
    y_pred_test = chosen_clf.predict(x_test)
    y_pred_proba_test = chosen_clf.predict_proba(x_test)
    loss = log_loss(y_test, y_pred_proba_test)
    print('Logistic Regression:')
    print('penalty = ', best_model['logistic__penalty'], 'C = ', best_model['logistic__C'], ', lambda = ',
          1 / best_model['logistic__C'])
    print('Loss = ', loss, '\n')

    # visualize the performances:
    clf_type = penalty
    plot_radar_logreg(clf, clf_type, lmbda)

    confusion_matrix(chosen_clf, x_test, y_test, y_pred_test, y_pred_proba_test)

    return chosen_clf, clf


def plot_radar_logreg(clf, clf_type, lmbda):
    """"
    :param clf: all the testing models.
    :param clf_type: the penalty that we used.
    :param lmbda: the different weights we checked.
    :return plot radar.
    -- taken from tutorial 5, written by Moran Davoodi with the assitance of Yuval Ben Sason & Kevin Kotzen -- 
    """""
    labels = np.array(['Accuracy', 'F1', 'PPV', 'Sensitivity', 'AUROC'])
    score_mat_train = np.stack((clf.cv_results_['mean_train_accuracy'], clf.cv_results_['mean_train_f1'],
                                clf.cv_results_['mean_train_precision'], clf.cv_results_['mean_train_recall'],
                                clf.cv_results_['mean_train_roc_auc']), axis=0)
    score_mat_val = np.stack((clf.cv_results_['mean_test_accuracy'], clf.cv_results_['mean_test_f1'],
                              clf.cv_results_['mean_test_precision'], clf.cv_results_['mean_test_recall'],
                              clf.cv_results_['mean_test_roc_auc']), axis=0)

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    # close the plot

    angles = np.concatenate((angles, [angles[0]]))
    cv_dict = clf.cv_results_['params']
    fig = plt.figure(figsize=(18, 14))
    for idx, loc in enumerate(cv_dict):
        ax = fig.add_subplot(1, len(lmbda), 1 + idx, polar=True)
        stats_train = score_mat_train[:, idx]
        stats_train = np.concatenate((stats_train, [stats_train[0]]))
        ax.plot(angles, stats_train, 'o-', linewidth=2)
        ax.fill(angles, stats_train, alpha=0.25)
        stats_val = score_mat_val[:, idx]
        stats_val = np.concatenate((stats_val, [stats_val[0]]))
        ax.plot(angles, stats_val, 'o-', linewidth=2)
        ax.fill(angles, stats_val, alpha=0.25)
        ax.set_thetagrids(angles[0:-1] * 180 / np.pi, labels)
        if idx == 0:
            ax.set_ylabel(clf_type, fontsize=18)
        if cv_dict[idx]['logistic__C'] <= 1:
            ax.set_title('$\lambda$ = %d' % (1 / cv_dict[idx]['logistic__C']))
        else:
            ax.set_title('$\lambda$ = %.3f' % (1 / cv_dict[idx]['logistic__C']))
        ax.set_ylim([0, 1])
        ax.legend(['Train', 'Validation'])
        ax.grid(True)
    plt.show()


def confusion_matrix(best_clf, x_test, y_test, y_pred_test, y_pred_proba_test):
    """""
    :param best_clf:the best model..
    :param x_test:  the testing dataset.
    :param y_test:  the testing labels. 
    :param y_pred_test: the predicted labels according to the chosen model/
    :param y_pred_proba_test: the probability of each labeling according to the best model
    :return plot_confusion_matrix & model's performance.
    """""

    from sklearn.metrics import plot_confusion_matrix, roc_auc_score
    from sklearn.metrics import confusion_matrix

    calc_TN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 0]
    calc_FP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[0, 1]
    calc_FN = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 0]
    calc_TP = lambda y_true, y_pred: confusion_matrix(y_true, y_pred)[1, 1]

    plot_confusion_matrix(best_clf, x_test, y_test, cmap=plt.cm.Blues)
    plt.grid(False)

    TP = calc_TP(y_test, y_pred_test)
    FN = calc_FN(y_test, y_pred_test)
    TN = calc_TN(y_test, y_pred_test)
    FP = calc_FP(y_test, y_pred_test)

    Se = TP / (TP + FN)
    Sp = TN / (TN + FP)
    PPV = TP / (TP + FP)
    NPV = TN / (FN + TN)
    Acc = (TP + TN) / (TP + TN + FP + FN)
    F1 = 2 * (PPV * Se) / (PPV + Se)

    print('Sensitivity is {:.2f}'.format(Se))
    print('Specificity is {:.2f}'.format(Sp))
    print('PPV is {:.2f}'.format(PPV))
    print('NPV is {:.2f}'.format(NPV))
    print('Accuracy is {:.2f}'.format(Acc))
    print('F1 is {:.2f}'.format(F1))
    print('AUROC is {:.2f}'.format(roc_auc_score(y_test, y_pred_proba_test[:, 1])))


def SVM_optimization(X_train, Y_train, x_test, y_test, degree=3, kernel='linear'):
    """""
    :param X_train = un-scaled X train dataset
    :param Y_train = un-scaled Y train dataset
    :param x_test  = un-scaled x test dataset
    :param y_test  = un-scaled y test dataset
    :param degree  = number of degrees for the non-linear svm
    :param kernel  = the svm type - 'linear' / 'non-linear'. 
                    if not linear, we will test 2 types of kernels -  'rbf', 'poly'
    """""
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, random_state=10, shuffle=True)
    svc = SVC(probability=True)
    C = np.array([0.01, 1, 10, 100])
    pipe = Pipeline(steps=[('scale', StandardScaler()), ('svm', svc)])
    if kernel == 'linear':
        svm = GridSearchCV(estimator=pipe,
                           param_grid={'svm__C': C, 'svm__kernel': [kernel]},
                           scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
                           cv=skf, refit='roc_auc', verbose=0, return_train_score=True)

    else:
        degree = np.array([degree])
        svm = GridSearchCV(estimator=pipe,
                           param_grid={'svm__kernel': ['rbf', 'poly'], 'svm__C': C,
                                       'svm__gamma': ['auto', 'scale'],
                                       'svm__degree': degree},
                           scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
                           cv=skf, refit='roc_auc', verbose=0, return_train_score=True)

    svm.fit(X_train, Y_train.squeeze())
    best_svm = svm.best_estimator_
    y_pred_test = best_svm.predict(x_test)
    y_pred_proba_test = best_svm.predict_proba(x_test)
    loss = hinge_loss(y_test, y_pred_test, labels=np.array([0, 1]))
    print('Support vector machine:')
    print('kernel = ', svm.best_params_['svm__kernel'], 'C = ', svm.best_params_['svm__C'])
    print('Loss = ', loss, '\n')

    # visualize the performances:
    if kernel == 'linear':
        clf_type = ['linear']
        plot_radar_svm(svm, clf_type)

    else:
        clf_type = ['rbf', 'scale']
        plot_radar_svm(svm, clf_type)
        clf_type = ['poly', 'scale']
        plot_radar_svm(svm, clf_type)

    confusion_matrix(best_svm, x_test, y_test, y_pred_test, y_pred_proba_test)

    return best_svm, svm


def plot_radar_svm(clf, clf_type):
    """"
    :param clf: all the testing models.
    :param clf_type: the kernel and gamma that we used.
    :return plot radar.
    -- taken from tutorial 5, written by Moran Davoodi with the assitance of Yuval Ben Sason & Kevin Kotzen -- 
    """""

    labels = np.array(['Accuracy', 'F1', 'PPV', 'Sensitivity', 'AUROC'])
    score_mat_train = np.stack((clf.cv_results_['mean_train_accuracy'], clf.cv_results_['mean_train_f1'],
                                clf.cv_results_['mean_train_precision'], clf.cv_results_['mean_train_recall'],
                                clf.cv_results_['mean_train_roc_auc']), axis=0)
    score_mat_val = np.stack((clf.cv_results_['mean_test_accuracy'], clf.cv_results_['mean_test_f1'],
                              clf.cv_results_['mean_test_precision'], clf.cv_results_['mean_test_recall'],
                              clf.cv_results_['mean_test_roc_auc']), axis=0)

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)

    angles = np.concatenate((angles, [angles[0]]))
    cv_dict = clf.cv_results_['params']
    fig = plt.figure(figsize=(18, 14))
    if 'svm__gamma' in cv_dict[0]:
        new_list = [(i, item) for i, item in enumerate(cv_dict) if
                    item["svm__kernel"] == clf_type[0] and item["svm__gamma"] == clf_type[1]]
    else:
        new_list = [(i, item) for i, item in enumerate(cv_dict) if
                    item["svm__kernel"] == clf_type[0]]
    for idx, val in enumerate(new_list):
        ax = fig.add_subplot(1, len(new_list), 1 + idx, polar=True)
        rel_idx, rel_dict = val
        stats_train = score_mat_train[:, rel_idx]
        stats_train = np.concatenate((stats_train, [stats_train[0]]))
        ax.plot(angles, stats_train, 'o-', linewidth=2)
        ax.fill(angles, stats_train, alpha=0.25)
        stats_val = score_mat_val[:, rel_idx]
        stats_val = np.concatenate((stats_val, [stats_val[0]]))
        ax.plot(angles, stats_val, 'o-', linewidth=2)
        ax.fill(angles, stats_val, alpha=0.25)
        ax.set_thetagrids(angles[0:-1] * 180 / np.pi, labels)
        if idx == 0:
            ax.set_ylabel(clf_type[0], fontsize=16)
        ax.set_title('C = %.3f' % (rel_dict['svm__C']))
        if 'svm__gamma' in cv_dict[0]:
            ax.set_xlabel('$\gamma = %s $' % (rel_dict['svm__gamma']))
        ax.set_ylim([0, 1])
        ax.legend(['Train', 'Validation'])
        ax.grid(True)

    plt.show()


def random_forest_optimization(X_train, Y_train, x_test, y_test):
    """"
    :param X_train: the training dataset.
    :param Y_train: the training labels.
    :param x_test  = un-scaled x test dataset
    :param y_test  = un-scaled y test dataset
    :return rfc - the random-forest model.
    """""

    from sklearn.ensemble import RandomForestClassifier

    rfc = RandomForestClassifier(max_depth=4, random_state=0, criterion='gini')
    rfc.fit(X_train, Y_train.squeeze())
    y_pred_test = rfc.predict(x_test)
    y_pred_proba_test = rfc.predict_proba(x_test)
    print('random forest results: ')
    loss = log_loss(y_test, y_pred_proba_test)
    print('Loss = ', loss, '\n')

    confusion_matrix(rfc, x_test, y_test, y_pred_test, y_pred_proba_test)

    return rfc


def plt_2d_pca(X_pca, y):
    """""
    :param X_pca: the dataset after reduced dimensionality
    :param y:     the labels fits to the x- dataset
    
    :return visualize the data reduced into 2D space using PCA
    
    reference: tutorial 09 - written by Moran Davoodi with the assitance of Alon Begin, Yuval Ben Sason & Kevin Kotzen
    """""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    ax.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color='b')
    ax.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color='r')
    ax.legend(('Negative', 'Positive'))
    ax.plot([0], [0], "ko")
    ax.arrow(0, 0, 0, 1, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    ax.arrow(0, 0, 1, 0, head_width=0.05, length_includes_head=True, head_length=0.1, fc='k', ec='k')
    ax.set_xlabel('$U_1$')
    ax.set_ylabel('$U_2$')
    ax.set_title('2D PCA')
    plt.show()


def PCA_transform(X_train, x_test, n_components=2):
    """""
    :param X_train: the training dataset 
    :param x_test:  the testing dataset
    :param n_components: the number of components we want to get.

    :return the datasets after reduced dimensionality
    """""

    pca = PCA(n_components=n_components, whiten=True)
    # apply PCA transformation
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(x_test)

    return X_train_pca, X_test_pca
