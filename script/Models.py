from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

def fit_logistic_regression(X_train, y_train, **options): ## TODO: Add other hyperparameters
    model = LogisticRegression(**options)
    model.fit(X_train, y_train)
    accuracy_train = model.score(X_train, y_train)
    return model, accuracy_train


def fit_decision_tree(X_train, y_train, **options): ## TODO: Add other hyperparameters
    model = DecisionTreeClassifier(**options)
    model.fit(X_train, y_train)
    accuracy_train = model.score(X_train, y_train)
    return model, accuracy_train


def fit_random_forest(X_train, y_train, **options): ## TODO: Add other hyperparameters
    model = RandomForestClassifier(**options)
    model.fit(X_train, y_train)
    accuracy_train = model.score(X_train, y_train)
    return model, accuracy_train


def fit_svm(X_train, y_train, **options): ## TODO: Add other hyperparameters
    model = SVC(**options)
    model.fit(X_train, y_train)
    accuracy_train = model.score(X_train, y_train)
    return model, accuracy_train


def fit_knn(X_train, y_train, **options): ## TODO: Add other hyperparameters
    model = KNeighborsClassifier(**options)
    model.fit(X_train, y_train)
    accuracy_train = model.score(X_train, y_train)
    return model, accuracy_train



######### MODEL FITTING ##########

def fit_logistic_regression(X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None):
    print("Log: Fitting Logistic Regression Model")
    model, train_accuracy = fit_logistic_regression(
        X_train, y_train, 
        penalty="l2", 
        multi_class='multinomial', #HYPERPARAMETER 
        max_iter=4000, 
        solver='lbfgs')
    
    print("Log: Train Accuracy: ", train_accuracy)
    if X_valid is not None and y_valid is not None:
        print("Log: Validation Accuracy: ", model.score(X_valid, y_valid))
    if X_test is not None and y_test is not None:
        print("Log: Test Accuracy: ", model.score(X_test, y_test))
    return model

def fit_decision_tree(X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None):
    print("Log: Fitting Decision Tree Model")
    model, train_accuracy = fit_decision_tree(
        X_train, y_train, 
        criterion="gini", 
        max_depth=100,  # HYPERPARAMETER
        min_samples_split=2, 
        min_samples_leaf=1)
    
    print("Log: Train Accuracy: ", train_accuracy)
    if X_valid is not None and y_valid is not None:
        print("Log: Validation Accuracy: ", model.score(X_valid, y_valid))
    if X_test is not None and y_test is not None:
        print("Log: Test Accuracy: ", model.score(X_test, y_test))
    return model

def fit_random_forest(X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None):
    print("Log: Fitting Random Forest Model")
    model, train_accuracy = fit_random_forest(
        X_train, y_train, 
        n_estimators=100, 
        criterion="gini", # HYPERPARAMETER
        max_depth=100, 
        min_samples_split=2, 
        min_samples_leaf=1)
    
    print("Log: Train Accuracy: ", train_accuracy)
    if X_valid is not None and y_valid is not None:
        print("Log: Validation Accuracy: ", model.score(X_valid, y_valid))
    if X_test is not None and y_test is not None:
        print("Log: Test Accuracy: ", model.score(X_test, y_test))
    return model

def fit_svm(X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None):
    print("Log: Fitting SVM Model")
    model, train_accuracy = fit_svm(
        X_train, y_train, 
        C=1.0, 
        kernel='rbf', 
        degree=3,  # HYPERPARAMETER
        gamma='scale', 
        coef0=0.0)
    
    print("Log: Train Accuracy: ", train_accuracy)
    if X_valid is not None and y_valid is not None:
        print("Log: Validation Accuracy: ", model.score(X_valid, y_valid))
    if X_test is not None and y_test is not None:
        print("Log: Test Accuracy: ", model.score(X_test, y_test))
    return model

def fit_knn(X_train, y_train, X_valid=None, y_valid=None, X_test=None, y_test=None):
    print("Log: Fitting KNN Model")
    model, train_accuracy = fit_knn(
        X_train, y_train, 
        n_neighbors=25, 
        weights='uniform', 
        algorithm='auto',  # HYPERPARAMETER  
        leaf_size=30, 
        p=2, 
        metric='minkowski')
    
    print("Log: Train Accuracy: ", train_accuracy)
    if X_valid is not None and y_valid is not None:
        print("Log: Validation Accuracy: ", model.score(X_valid, y_valid))
    if X_test is not None and y_test is not None:
        print("Log: Test Accuracy: ", model.score(X_test, y_test))
    return model