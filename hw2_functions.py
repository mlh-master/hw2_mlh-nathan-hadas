import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib as path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import pickle
import sys
import matplotlib as mpl

mpl.style.use(['ggplot'])
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# directory_Nathan = r'C:\Users\Nathan\PycharmProjects\ML_in_Healthcare_Winter2021\HW2'
directory_Hadas = r'C:\Users\hadas\Documents\OneDrive - Technion\semester_7\Machine learning in healthcare\Hw\HW2'
# file_path = path.cwd().joinpath('HW2_data.csv')
file_path = os.path.join(directory_Hadas, 'HW2_data.csv')
file = pd.read_csv(file_path)

# Todo: clean nan data:

missing_data = file[file.isnull().any(axis=1)]
pos_nan = missing_data[missing_data['Diagnosis'] == 'Positive']
positive_val = file[file['Diagnosis'] == 'Positive']

# Our data consists mainly of 'Positive' samples, thus in order to reduce unnecessary/
# data distortion, we decided to remove missing data labeled 'Positive'

clean_data = file.drop(pos_nan.index)

# All nan samples are now necessarily labeled as 'Negative'

# We decided to complete the missing samples by the distribution of each/
# feature for samples tagged as a 'Negative'

neg_data = clean_data[clean_data['Diagnosis'] == 'Negative']
for key in neg_data.keys():
    temp_prob = neg_data[key].value_counts()
    dominant_val = temp_prob[temp_prob == max(temp_prob)].index[0]

    clean_data[key] = [val if not pd.isna(val) else dominant_val for val in clean_data[key]]

# Todo: convert the data to one-hot vector:
#   we will ignore the 'Age' category because it already numerical.

Age = clean_data['Age'].to_numpy()
X_before_encode = clean_data.drop(['Diagnosis', 'Age'], axis=1)
y = clean_data['Diagnosis'].to_numpy().reshape(-1, 1)
enc = OneHotEncoder(drop='if_binary')
X_before_encode = enc.fit_transform(X_before_encode).toarray()
y = enc.fit_transform(y).toarray()

X = np.empty([X_before_encode.shape[0], X_before_encode.shape[1] + 1])
X[:, 0] = Age
X[:, 1:] = X_before_encode

# Todo: split data function: split the data randomly to train-test with test fraction

X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)

# Todo: Provide a detailed visualization and exploration of the data
#   show the diagnosis distribution:
Y = clean_data['Diagnosis']
Y.value_counts().plot(kind="pie", labels=['Positive', 'Negative'], colors=['steelblue', 'salmon'], autopct='%1.1f%%')
plt.show()


# Todo: compare between test & training dataset

def table_plot(X_train, x_test, Y_train, y_test, clean_data):
    positive_features = clean_data.keys()
    table_dict = {}
    idx = 0
    for feat in positive_features:
        if feat == 'Age':
            avg_Age_train = round((X_train[:, idx].mean()))
            avg_Age_test = round((x_test[:, idx].mean()))
            delta = avg_Age_train - avg_Age_test
            table_dict[feat] = [avg_Age_train, avg_Age_test, delta]
            idx += 1
        elif feat == 'Diagnosis':
            avg_diagnosis_train = round(Y_train.mean() * 100)
            avg_diagnosis_test = round(y_test.mean() * 100)
            delta = avg_diagnosis_train - avg_diagnosis_test
            table_dict[feat] = [avg_diagnosis_train, avg_diagnosis_test, delta]
        else:
            avg_train = round(X_train[:, idx].mean() * 100)
            avg_test = round(x_test[:, idx].mean() * 100)
            delta = avg_train - avg_test
            table_dict[feat] = [avg_train, avg_test, delta]
            idx += 1

    columns = ['Train %', 'Test %', 'Delta %']
    rows = list(table_dict.keys())
    cells = list(table_dict.values())

    plt.figure()
    table = plt.table(cellText=cells, rowLabels=rows, colLabels=columns)
    plt.axis("off")
    plt.grid(False)
    plt.show()


table_plot(X_train, x_test, Y_train, y_test, clean_data)


# Todo: create bar plot for the specified filed:

def bar_plot(clean_data, feature):
    feature_data = clean_data[[feature, 'Diagnosis']]
    sns.set_theme(style="darkgrid")
    ax = sns.countplot(x=feature, hue="Diagnosis", data=feature_data)
    plt.show()


features = ['Gender', 'Weakness', 'Increased Thirst', 'Obesity']
for feature in features:
    bar_plot(clean_data, feature)

# Todo: plot data distribution:
sns.pairplot(clean_data.loc[:, 'Age':'Diagnosis'], hue="Diagnosis")
sns.displot(clean_data, x="Age", hue="Diagnosis")
plt.show()


# Todo: Training Logistic Regression:
#  use functions from previously tutorials

def check_penalty(penalty='none'):
    if penalty == 'l1':
        solver = 'liblinear'
    if penalty == 'l2' or penalty == 'none':
        solver = 'lbfgs'
    return solver


# use GridSearchCV to find the best logistic regression parameters:
# check for a list of different penalties:

def logReg_optimization(penalty, X_train, Y_train, x_test, y_test):
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, random_state=10, shuffle=True)
    max_iter = 2000
    solver = check_penalty(penalty)
    log_reg = LogisticRegression(random_state=5, max_iter=max_iter, solver=solver)
    lmbda = np.array([0.01, 0.01, 1, 10, 100, 1000])

    # Todo - ask if need to scale - ('scale', StandardScaler()),
    pipe = Pipeline(steps=[('logistic', log_reg)])
    clf = GridSearchCV(estimator=pipe, param_grid={'logistic__C': 1 / lmbda, 'logistic__penalty': [penalty]},
                       scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'], cv=skf,
                       refit='roc_auc', verbose=3, return_train_score=True)

    clf.fit(X_train, Y_train.squeeze())
    best_model = clf.best_params_

    # visualize the performances:

    params = ['C', 'penalty']
    clf_type = 'log_reg'
    plot_radar_logreg(clf, params, clf_type, lmbda)

    # Todo: Ask if need to scale!
    # scaler = StandardScaler()
    # x_train_scaled = scaler.fit_transform(X_train)
    # x_test_scaled = scaler.transform(x_test)

    chosen_clf = clf.best_estimator_
    y_pred_test = chosen_clf.predict(x_test)  # NOTICE NOT TO USE THE STANDARDIZED DATA.
    y_pred_proba_test = chosen_clf.predict_proba(x_test)

    loss = log_loss(y_test, y_pred_proba_test)
    print('Logistic Regression loss with the parameters: \n')
    print('penalty: ', best_model['logistic__penalty'], 'C: ', best_model['logistic__C'])
    print('Loss = ', loss)
    confusion_matrix(chosen_clf, x_test, y_test, y_pred_test, y_pred_proba_test)

    return chosen_clf, clf


def plot_radar_logreg(clf, params, clf_type, lmbda):
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
        ax.set_thetagrids(angles[0:-1] * 180 / np.pi, labels)  # noticed the fix!
        if idx == 0:
            ax.set_ylabel('$L_2$', fontsize=18)
        if cv_dict[idx]['logistic__C'] <= 1:
            ax.set_title('$\lambda$ = %d' % (1 / cv_dict[idx]['logistic__C']))
        else:
            ax.set_title('$\lambda$ = %.3f' % (1 / cv_dict[idx]['logistic__C']))
        ax.set_ylim([0, 1])
        ax.legend(['Train', 'Validation'])
        ax.grid(True)
    plt.show()


def confusion_matrix(best_clf, x_test, y_test, y_pred_test, y_pred_proba_test):
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


best_model_l1, clf_l1 = logReg_optimization('l1', X_train, Y_train, x_test, y_test)
best_model_l2, clf_l2 = logReg_optimization('l2', X_train, Y_train, x_test, y_test)


# Todo: Support vector machine

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
    C = np.array([0.001, 0.01, 1, 10, 100, 1000])
    pipe = Pipeline(steps=[('scale', StandardScaler()), ('svm', svc)])
    if kernel == 'linear':
        svm = GridSearchCV(estimator=pipe,
                           param_grid={'svm__C': C, 'svm__kernel': [kernel]},
                           scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
                           cv=skf, refit='roc_auc', verbose=3, return_train_score=True)

    else:
        degree = np.array([degree])
        svm = GridSearchCV(estimator=pipe,
                           param_grid={'svm__kernel': ['rbf', 'poly'], 'svm__C': C,
                                       'svm__gamma': ['auto', 'scale'],
                                       'svm__degree': degree},
                           scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc'],
                           cv=skf, refit='roc_auc', verbose=3, return_train_score=True)

    svm.fit(X_train, Y_train.squeeze())
    best_svm = svm.best_estimator_
    print(svm.best_params_)

    # visualize the performances:
    if kernel == 'linear':
        clf_type = ['linear']
        plot_radar_svm(svm, clf_type)

    else:
        clf_type = ['rbf', 'scale']
        plot_radar_svm(svm, clf_type)
        clf_type = ['poly', 'scale']
        plot_radar_svm(svm, clf_type)

    y_pred_test = best_svm.predict(x_test)
    y_pred_proba_test = best_svm.predict_proba(x_test)

    confusion_matrix(best_svm, x_test, y_test, y_pred_test, y_pred_proba_test)

    return best_svm, svm


def plot_radar_svm(clf, clf_type):
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
            ax.set_ylabel(clf_type[0], fontsize=18)
        ax.set_title('C = %.3f' % (rel_dict['svm__C']))
        if 'svm__gamma' in cv_dict[0]:
            ax.set_xlabel('$\gamma = %s $' % (rel_dict['svm__gamma']))
        ax.set_ylim([0, 1])
        ax.legend(['Train', 'Validation'])
        ax.grid(True)

    plt.show()


best_svm_linear, svm_linear = SVM_optimization(X_train, Y_train, x_test, y_test, degree=0, kernel='linear')
best_svm_nonlinear, svm_nonlinear = SVM_optimization(X_train, Y_train, x_test, y_test, degree=3, kernel='nonlinear')


# Todo: random forest:
def random_forest_optimization(X_train, Y_train, x_test, y_test):
    from sklearn.ensemble import RandomForestClassifier

    rfc = RandomForestClassifier(max_depth=4, random_state=0, criterion='gini')
    rfc.fit(X_train, Y_train.squeeze())
    y_pred_test = rfc.predict(x_test)
    y_pred_proba_test = rfc.predict_proba(x_test)
    confusion_matrix(rfc, x_test, y_test, y_pred_test, y_pred_proba_test)

    return rfc


rfc = random_forest_optimization(X_train, Y_train, x_test, y_test)

# Todo: find the added value of each feature from the random forest:
#  reference: https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
importance = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
indices = np.argsort(importance)[::-1]
# Print the feature ranking
print("Feature ranking:")

keys_exc_diagnosis = clean_data.drop(['Diagnosis'], axis=1)
keys_list = list(keys_exc_diagnosis.keys())
key_sort = []
for f in range(X_train.shape[1]):
    key_sort += [keys_list[indices[f]]]
    print(key_sort[f], " %d (%f)" % (indices[f], importance[indices[f]]))

# Plot the impurity-based feature importances of the forest
plt.figure(figsize=(60, 30))
plt.title("Feature importance")
data = {'keys': key_sort, 'imp': importance[indices]}
ax = sns.barplot(x='keys', y="imp", data=data)
plt.show()

# Todo: compare between different classifiers

from sklearn.metrics import roc_auc_score, plot_roc_curve

classifiers = [best_model_l1, best_model_l2, best_svm_linear, best_svm_nonlinear, rfc]
roc_score = []
plt.figure()
ax = plt.gca()
for clf in classifiers:
    plot_roc_curve(clf, x_test, y_test, ax=ax)
    roc_score.append(np.round_(roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1]), decimals=3))
ax.plot(np.linspace(0, 1, x_test.shape[0]), np.linspace(0, 1, x_test.shape[0]))
plt.legend(('log_reg - l1, AUROC = ' + str(roc_score[0]), 'log_reg - l2, AUROC = ' + str(roc_score[1]),
            'lin_svm, AUROC = ' + str(roc_score[2]), 'nonlin_svm, AUROC = ' + str(roc_score[3]),
            'rfc, AUROC = ' + str(roc_score[4]), 'flipping a coin'))
plt.show()


# Todo: Data Separability Visualization

from sklearn.decomposition import PCA

def plt_2d_pca(X_pca, y):
    """""
    :param X_pca
    :param y
    
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
    pca = PCA(n_components=n_components, whiten=True)
    # apply PCA transformation
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(x_test)

    return X_train_pca, X_test_pca


X_train_pca, X_test_pca = PCA_transform(X_train, x_test, n_components=2)
plt_2d_pca(X_test_pca, y_test.squeeze())


# Todo: train the models on the dimension - reduced trainig set:
best_model_l1_pca, clf_l1_pca = logReg_optimization('l1', X_train_pca, Y_train, X_test_pca, y_test)
best_model_l2_pca, clf_l2_pca = logReg_optimization('l2', X_train_pca, Y_train, X_test_pca, y_test)

best_svm_linear_pca, svm_linear_pca = SVM_optimization(X_train_pca, Y_train, X_test_pca, y_test, degree=0, kernel='linear')
best_svm_nonlinear_pca, svm_nonlinear_pca = SVM_optimization(X_train_pca, Y_train, X_test_pca, y_test, degree=3, kernel='nonlinear')

rfc_pac = random_forest_optimization(X_train_pca, Y_train, X_test_pca, y_test)


classifiers = [best_model_l1_pca, best_model_l2_pca, best_svm_linear_pca, best_svm_nonlinear_pca, rfc_pac]
roc_score = []
plt.figure()
ax = plt.gca()
for clf in classifiers:
    plot_roc_curve(clf, X_test_pca, y_test, ax=ax)
    roc_score.append(np.round_(roc_auc_score(y_test, clf.predict_proba(X_test_pca)[:, 1]), decimals=3))
ax.plot(np.linspace(0, 1, X_test_pca.shape[0]), np.linspace(0, 1, X_test_pca.shape[0]))
plt.legend(('log_reg - l1, AUROC = ' + str(roc_score[0]), 'log_reg - l2, AUROC = ' + str(roc_score[1]),
            'lin_svm, AUROC = ' + str(roc_score[2]), 'nonlin_svm, AUROC = ' + str(roc_score[3]),
            'rfc, AUROC = ' + str(roc_score[4]), 'flipping a coin'))
plt.show()


# Todo: train the models on the 2 selected features:
X_train_2feat = X_train[:, indices[0:2]]
X_test_2feat = x_test[:, indices[0:2]]

best_model_l1_2feat, clf_l1_2feat = logReg_optimization('l1', X_train_2feat, Y_train, X_test_2feat, y_test)
best_model_l2_2feat, clf_l2_2feat = logReg_optimization('l2', X_train_2feat, Y_train, X_test_2feat, y_test)

best_svm_linear_2feat, svm_linear_2feat = SVM_optimization(X_train_2feat, Y_train, X_test_2feat, y_test, degree=0, kernel='linear')
best_svm_nonlinear_2feat, svm_nonlinear_2feat = SVM_optimization(X_train_2feat, Y_train, X_test_2feat, y_test, degree=3,
                                                                 kernel='nonlinear')

rfc_2feat = random_forest_optimization(X_train_2feat, Y_train, X_test_2feat, y_test)


classifiers = [best_model_l1_2feat, best_model_l2_2feat, best_svm_linear_2feat, best_svm_nonlinear_2feat, rfc_2feat]
roc_score = []
plt.figure()
ax = plt.gca()
for clf in classifiers:
    plot_roc_curve(clf, X_test_2feat, y_test, ax=ax)
    roc_score.append(np.round_(roc_auc_score(y_test, clf.predict_proba(X_test_2feat)[:, 1]), decimals=3))
ax.plot(np.linspace(0, 1, X_test_2feat.shape[0]), np.linspace(0, 1, X_test_2feat.shape[0]))
plt.legend(('log_reg - l1, AUROC = ' + str(roc_score[0]), 'log_reg - l2, AUROC = ' + str(roc_score[1]),
            'lin_svm, AUROC = ' + str(roc_score[2]), 'nonlin_svm, AUROC = ' + str(roc_score[3]),
            'rfc, AUROC = ' + str(roc_score[4]), 'flipping a coin'))
plt.show()
