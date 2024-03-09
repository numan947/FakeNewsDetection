from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
import numpy as np
import time
from sklearn.utils import class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

def fit_extra_trees_classifier(X_train, y_train, **options): ## TODO: Add other hyperparameters
    model = ExtraTreesClassifier(**options)
    model.fit(X_train, y_train)
    return model


def fit_adaboost_classifier(X_train, y_train, **options): ## TODO: Add other hyperparameters
    model = AdaBoostClassifier(**options)
    model.fit(X_train, y_train)
    return model

def fit_xgboost_classifier(X_train, y_train, **options): ## TODO: Add other hyperparameters
    model = xgb.XGBClassifier(**options)
    model.fit(X_train, y_train)
    return model

def fit_gaussian_nb_classifier(X_train, y_train, **options): ## TODO: Add other hyperparameters
    model = GaussianNB(**options)
    model.fit(X_train, y_train)
    return model

def fit_logistic_regression_classifier(X_train, y_train, **options): ## TODO: Add other hyperparameters
    model = LogisticRegression(**options)
    model.fit(X_train, y_train)
    return model


def fit_decision_tree_classifier(X_train, y_train, **options): ## TODO: Add other hyperparameters
    model = DecisionTreeClassifier(**options)
    model.fit(X_train, y_train)
    return model


def fit_random_forest_classifier(X_train, y_train, **options): ## TODO: Add other hyperparameters
    model = RandomForestClassifier(**options)
    model.fit(X_train, y_train)
    return model


def fit_svm_classifier(X_train, y_train, **options): ## TODO: Add other hyperparameters
    model = SVC(**options)
    model.fit(X_train, y_train)
    return model


def fit_knn_classifier(X_train, y_train, **options): ## TODO: Add other hyperparameters
    model = KNeighborsClassifier(**options)
    model.fit(X_train, y_train)
    return model

def get_metrics(model,X_train, y_train, X_valid = None, y_valid = None, X_test=None, y_test=None):
    y_train_pred = model.predict(X_train)
    y_train_pred_proba = model.predict_proba(X_train)
    y_valid_pred = None
    y_test_pred = None
    if X_valid is not None and y_valid is not None:
        y_valid_pred = model.predict(X_valid)
        y_valid_pred_proba = model.predict_proba(X_valid)
    if X_test is not None and y_test is not None:
        y_test_pred = model.predict(X_test)
        y_test_pred_proba = model.predict_proba(X_test)
        
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    train_auc = roc_auc_score(y_train, y_train_pred_proba, average='weighted', multi_class='ovr')
    
    valid_accuracy = None
    valid_precision = None
    valid_recall = None
    valid_f1 = None
    valid_auc = None
    if X_valid is not None and y_valid is not None:
        valid_accuracy = accuracy_score(y_valid, y_valid_pred)
        valid_precision = precision_score(y_valid, y_valid_pred, average='weighted')
        valid_recall = recall_score(y_valid, y_valid_pred, average='weighted')
        valid_f1 = f1_score(y_valid, y_valid_pred, average='weighted')
        valid_auc = roc_auc_score(y_valid, y_valid_pred_proba, average='weighted', multi_class='ovr')
    
    test_accuracy = None
    test_precision = None
    test_recall = None
    test_f1 = None
    test_auc = None
    if X_test is not None and y_test is not None:
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        test_auc = roc_auc_score(y_test, y_test_pred_proba, average='weighted', multi_class='ovr')
    
    # [
        # train_accuracy, valid_accuracy, test_accuracy, 
        # train_precision, valid_precision, test_precision, 
        # train_recall, valid_recall, test_recall, 
        # train_f1, valid_f1, test_f1, 
        # train_auc, valid_auc, test_auc
    # ]
    
    all_metrics = [
        train_accuracy, 
        valid_accuracy, 
        test_accuracy,
        train_precision,
        valid_precision,
        test_precision,
        train_recall,
        valid_recall,
        test_recall,
        train_f1,
        valid_f1,
        test_f1,
        train_auc,
        valid_auc,
        test_auc
        ]
    return all_metrics

######### MODEL FITTING ##########

def fit_logistic_regression(X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None):
    print("Log: Fitting Logistic Regression Model")
    train_time = time.time()
    model = fit_logistic_regression_classifier(
        X_train, y_train, 
        penalty="l2", 
        multi_class='multinomial', #HYPERPARAMETER 
        max_iter=4000, 
        solver='lbfgs')
    print("Log: Training Time: {:.3f} seconds".format(time.time()-train_time))
    all_metrics = get_metrics(model, X_train, y_train, X_valid, y_valid, X_test, y_test)
    train_accuracy, valid_accuracy, test_accuracy = all_metrics[:3]
    print("Log: Train Accuracy: ", train_accuracy)
    if X_valid is not None and y_valid is not None:
        print("Log: Validation Accuracy: ", valid_accuracy)
    if X_test is not None and y_test is not None:
        print("Log: Test Accuracy: ", test_accuracy)
    return model, all_metrics
    

def fit_decision_tree(X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None):
    print("Log: Fitting Decision Tree Model")
    train_time = time.time()
    model = fit_decision_tree_classifier(
        X_train, y_train, 
        criterion="gini", 
        max_depth=100,  # HYPERPARAMETER
        min_samples_split=2, 
        min_samples_leaf=1)
    
    print("Log: Training Time: {:.3f} seconds".format(time.time()-train_time))
    all_metrics = get_metrics(model, X_train, y_train, X_valid, y_valid, X_test, y_test)
    train_accuracy, valid_accuracy, test_accuracy = all_metrics[:3]
    print("Log: Train Accuracy: ", train_accuracy)
    if X_valid is not None and y_valid is not None:
        print("Log: Validation Accuracy: ", valid_accuracy)
    if X_test is not None and y_test is not None:
        print("Log: Test Accuracy: ", test_accuracy)
    return model, all_metrics

def fit_random_forest(X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None):
    print("Log: Fitting Random Forest Model")
    train_time = time.time()
    model = fit_random_forest_classifier(
        X_train, y_train, 
        n_estimators=100, 
        criterion="gini", # HYPERPARAMETER
        max_depth=100, 
        min_samples_split=2, 
        min_samples_leaf=1)
    
    print("Log: Training Time: {:.3f} seconds".format(time.time()-train_time))
    all_metrics = get_metrics(model, X_train, y_train, X_valid, y_valid, X_test, y_test)
    train_accuracy, valid_accuracy, test_accuracy = all_metrics[:3]
    print("Log: Train Accuracy: ", train_accuracy)
    if X_valid is not None and y_valid is not None:
        print("Log: Validation Accuracy: ", valid_accuracy)
    if X_test is not None and y_test is not None:
        print("Log: Test Accuracy: ", test_accuracy)
    return model, all_metrics

def fit_svm(X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None):
    print("Log: Fitting SVM Model")
    train_time = time.time()
    model = fit_svm_classifier(
        X_train, y_train,
        probability=True,
        C=1.0, 
        kernel='rbf', 
        degree=3,  # HYPERPARAMETER
        gamma='scale', 
        coef0=0.0)
    
    print("Log: Training Time: {:.3f} seconds".format(time.time()-train_time))
    all_metrics = get_metrics(model, X_train, y_train, X_valid, y_valid, X_test, y_test)
    train_accuracy, valid_accuracy, test_accuracy = all_metrics[:3]
    print("Log: Train Accuracy: ", train_accuracy)
    if X_valid is not None and y_valid is not None:
        print("Log: Validation Accuracy: ", valid_accuracy)
    if X_test is not None and y_test is not None:
        print("Log: Test Accuracy: ", test_accuracy)
    return model, all_metrics

def fit_knn(X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None):
    print("Log: Fitting KNN Model")
    train_time = time.time()
    model = fit_knn_classifier(
        X_train, y_train, 
        n_neighbors=25, 
        weights='uniform', 
        algorithm='auto',  # HYPERPARAMETER  
        leaf_size=30, 
        p=2, 
        metric='minkowski')
    print("Log: Training Time: {:.3f} seconds".format(time.time()-train_time))
    all_metrics = get_metrics(model, X_train, y_train, X_valid, y_valid, X_test, y_test)
    train_accuracy, valid_accuracy, test_accuracy = all_metrics[:3]
    
    print("Log: Train Accuracy: ", train_accuracy)
    if X_valid is not None and y_valid is not None:
        print("Log: Validation Accuracy: ", valid_accuracy)
    if X_test is not None and y_test is not None:
        print("Log: Test Accuracy: ", test_accuracy)
    return model, all_metrics

def fit_gaussian_nb(X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None):
    print("Log: Fitting Gaussian NB Model")
    train_time = time.time()
    model = fit_gaussian_nb_classifier(X_train.toarray(), y_train)
    print("Log: Training Time: {:.3f} seconds".format(time.time()-train_time))
    # This is a sparse matrix, so we need to convert it to dense matrix for GaussianNB
    all_metrics = get_metrics(model, X_train.toarray(), y_train, X_valid.toarray(), y_valid, X_test.toarray(), y_test)
    train_accuracy, valid_accuracy, test_accuracy = all_metrics[:3]
    print("Log: Train Accuracy: ", train_accuracy)
    if X_valid is not None and y_valid is not None:
        print("Log: Validation Accuracy: ", valid_accuracy)
    if X_test is not None and y_test is not None:
        print("Log: Test Accuracy: ", test_accuracy)
    return model, all_metrics

def fit_adaboost(X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None):
    print("Log: Fitting AdaBoost Model")
    train_time = time.time()
    model = fit_adaboost_classifier(
        X_train, y_train, 
        n_estimators=300, 
        learning_rate=1.0,
        algorithm='SAMME') # HYPERPARAMETER
    print("Log: Training Time: {:.3f} seconds".format(time.time()-train_time))
    all_metrics = get_metrics(model, X_train, y_train, X_valid, y_valid, X_test, y_test)
    train_accuracy, valid_accuracy, test_accuracy = all_metrics[:3]
    print("Log: Train Accuracy: ", train_accuracy)
    if X_valid is not None and y_valid is not None:
        print("Log: Validation Accuracy: ", valid_accuracy)
    if X_test is not None and y_test is not None:
        print("Log: Test Accuracy: ", test_accuracy)
    return model, all_metrics

def fit_extra_trees(X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None):
    print("Log: Fitting Extra Trees Model")
    train_time = time.time()
    model = fit_extra_trees_classifier(
        X_train, y_train, 
        n_estimators=300, 
        criterion='gini', 
        max_depth=23, 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0.0, 
        max_features=500, # HYPERPARAMETER 
        max_leaf_nodes=None, 
        bootstrap=False, 
        oob_score=False, 
        n_jobs=None, 
        random_state=None, 
        verbose=0, 
        warm_start=False, 
        ccp_alpha=0.0, 
        max_samples=None)
    print("Log: Training Time: {:.3f} seconds".format(time.time()-train_time))
    all_metrics = get_metrics(model, X_train, y_train, X_valid, y_valid, X_test, y_test)
    train_accuracy, valid_accuracy, test_accuracy = all_metrics[:3]
    print("Log: Train Accuracy: ", train_accuracy)
    if X_valid is not None and y_valid is not None:
        print("Log: Validation Accuracy: ", valid_accuracy)
    if X_test is not None and y_test is not None:
        print("Log: Test Accuracy: ", test_accuracy)
    return model, all_metrics

def fit_xgboost(X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None, class_mapping=None):
    print("Log: Fitting XGBoost Model")
    y_num_train = [class_mapping[label] for label in y_train]
    y_num_valid = [class_mapping[label] for label in y_valid]
    y_num_test = [class_mapping[label] for label in y_test] 
    
    train_time = time.time()
    model = fit_xgboost_classifier(
        X_train, y_num_train,
        booster='gbtree',
        objective='multi:softmax',
        num_class=len(set(y_num_train)),
        eval_metric='auc',
        n_estimators=500,
        max_depth=10, # HYPERPARAMETER
        subsample=0.45,
        colsample_bytree=0.45,
    )
    print("Log: Training Time: {:.3f} seconds".format(time.time()-train_time))
    
    all_metrics = get_metrics(model, X_train, y_num_train, X_valid, y_num_valid, X_test, y_num_test)
    train_accuracy, valid_accuracy, test_accuracy = all_metrics[:3]
    print("Log: Train Accuracy: ", train_accuracy)
    if X_valid is not None and y_valid is not None:
        print("Log: Validation Accuracy: ", valid_accuracy)
    if X_test is not None and y_test is not None:
        print("Log: Test Accuracy: ", test_accuracy)
    return model, all_metrics