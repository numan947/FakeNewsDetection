from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb

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
    train_accuracy = model.score(X_train, y_train)
    valid_accuracy = None
    test_accuracy = None
    if X_valid is not None and y_valid is not None:
        valid_accuracy = model.score(X_valid, y_valid)
    if X_test is not None and y_test is not None:
        test_accuracy = model.score(X_test, y_test)
    
    # [
        # train_accuracy, valid_accuracy, test_accuracy, 
        # train_precision, valid_precision, test_precision, 
        # train_recall, valid_recall, test_recall, 
        # train_f1, valid_f1, test_f1, 
        # train_auc, valid_auc, test_auc
    # ]
    
    all_metrics = [train_accuracy, valid_accuracy, test_accuracy]
    return all_metrics

######### MODEL FITTING ##########

def fit_logistic_regression(X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None):
    print("Log: Fitting Logistic Regression Model")
    model = fit_logistic_regression_classifier(
        X_train, y_train, 
        penalty="l2", 
        multi_class='multinomial', #HYPERPARAMETER 
        max_iter=4000, 
        solver='lbfgs')
    
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
    model = fit_decision_tree_classifier(
        X_train, y_train, 
        criterion="gini", 
        max_depth=100,  # HYPERPARAMETER
        min_samples_split=2, 
        min_samples_leaf=1)
    
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
    model = fit_random_forest_classifier(
        X_train, y_train, 
        n_estimators=100, 
        criterion="gini", # HYPERPARAMETER
        max_depth=100, 
        min_samples_split=2, 
        min_samples_leaf=1)
    
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
    model = fit_svm_classifier(
        X_train, y_train, 
        C=1.0, 
        kernel='rbf', 
        degree=3,  # HYPERPARAMETER
        gamma='scale', 
        coef0=0.0)
    
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
    model = fit_knn_classifier(
        X_train, y_train, 
        n_neighbors=25, 
        weights='uniform', 
        algorithm='auto',  # HYPERPARAMETER  
        leaf_size=30, 
        p=2, 
        metric='minkowski')
    
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
    model = fit_gaussian_nb_classifier(X_train.toarray(), y_train)
    
    print("Log: Train Accuracy: ", train_accuracy)
    if X_valid is not None and y_valid is not None:
        print("Log: Validation Accuracy: ", model.score(X_valid.toarray(), y_valid))
    if X_test is not None and y_test is not None:
        print("Log: Test Accuracy: ", model.score(X_test.toarray(), y_test))
    return model

def fit_adaboost(X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None):
    print("Log: Fitting AdaBoost Model")
    model = fit_adaboost_classifier(
        X_train, y_train, 
        n_estimators=50, 
        learning_rate=1.0, 
        algorithm='SAMME.R') # HYPERPARAMETER
    
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
    model = fit_extra_trees_classifier(
        X_train, y_train, 
        n_estimators=100, 
        criterion='gini', 
        max_depth=100, 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0.0, 
        max_features='auto', 
        max_leaf_nodes=None, 
        min_impurity_decrease=0.0, 
        min_impurity_split=None, 
        bootstrap=False, 
        oob_score=False, 
        n_jobs=None, 
        random_state=None, 
        verbose=0, 
        warm_start=False, 
        ccp_alpha=0.0, 
        max_samples=None)
    
    all_metrics = get_metrics(model, X_train, y_train, X_valid, y_valid, X_test, y_test)
    train_accuracy, valid_accuracy, test_accuracy = all_metrics[:3]
    print("Log: Train Accuracy: ", train_accuracy)
    if X_valid is not None and y_valid is not None:
        print("Log: Validation Accuracy: ", valid_accuracy)
    if X_test is not None and y_test is not None:
        print("Log: Test Accuracy: ", test_accuracy)
    return model, all_metrics

def fit_xgboost(X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None):
    print("Log: Fitting XGBoost Model")
    model = fit_xgboost_classifier(
        X_train, y_train, 
        objective='binary:logistic', 
        use_label_encoder=True, 
        base_score=0.5, 
        booster='gbtree', 
        colsample_bylevel=1, 
        colsample_bynode=1, 
        colsample_bytree=1, 
        gamma=0, 
        gpu_id=-1, 
        importance_type='gain', 
        interaction_constraints='', 
        learning_rate=0.300000012, 
        max_delta_step=0, 
        max_depth=6, 
        min_child_weight=1, 
        missing=None, 
        monotone_constraints='()', 
        n_estimators=100, 
        n_jobs=8, 
        num_parallel_tree=1, 
        random_state=0, 
        reg_alpha=0, 
        reg_lambda=1, 
        scale_pos_weight=1, 
        subsample=1, 
        tree_method='exact', 
        validate_parameters=1, 
        verbosity=None)
    
    all_metrics = get_metrics(model, X_train, y_train, X_valid, y_valid, X_test, y_test)
    train_accuracy, valid_accuracy, test_accuracy = all_metrics[:3]
    print("Log: Train Accuracy: ", train_accuracy)
    if X_valid is not None and y_valid is not None:
        print("Log: Validation Accuracy: ", valid_accuracy)
    if X_test is not None and y_test is not None:
        print("Log: Test Accuracy: ", test_accuracy)
    return model, all_metrics