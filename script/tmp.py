def do_tfidf(train_data, valid_data, test_data, metrics_data):

    # 2. GLOVE.6B.100d
    print("\n\nLog: Extracting GLOVE.6B.100d")
    FIGURES = "../generated/figures/liar_data/tfidf/"
    if os.path.exists(FIGURES) == False:
        os.makedirs(FIGURES)
 
    X_train, y_train, X_valid, y_valid, X_test, y_test = extract_tfidf(train_data, valid_data, test_data)
   
    model, metrics = Models.fit_logistic_regression(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["Logistic Regression", "GLOVE.6B.100d"] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "Logistic Regression", FIGURES+"logistic_regression")
    
    model, metrics = Models.fit_decision_tree(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["Decision Tree", "GLOVE.6B.100d"] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "Decision Tree", FIGURES+"decision_tree")
    
    model, metrics = Models.fit_random_forest(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["Random Forest", "GLOVE.6B.100d"] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "Random Forest", FIGURES+"random_forest")
    
    model, metrics = Models.fit_svm(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["SVM", "GLOVE.6B.100d"] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "SVM", FIGURES+"svm")
    
    model, metrics = Models.fit_knn(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["KNN", "GLOVE.6B.100d"] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "KNN", FIGURES+"knn")
    
    model, metrics = Models.fit_gaussian_nb(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["Gaussian NB", "GLOVE.6B.100d"] + metrics)
    create_confusion_matrix(model, X_train.toarray(), y_train, X_valid.toarray(), y_valid, X_test.toarray(), y_test, "Gaussian NB", FIGURES+"gaussian_nb")
    
    model, metrics = Models.fit_adaboost(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["AdaBoost", "GLOVE.6B.100d"] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "AdaBoost", FIGURES+"adaboost")
    
    model, metrics = Models.fit_extra_trees(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["Extra Trees", "GLOVE.6B.100d"] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "Extra Trees", FIGURES+"extra_trees")
    
    model, metrics = Models.fit_xgboost(X_train, y_train, X_valid, y_valid, X_test, y_test, class_mapping)
    metrics_data.append(["XGBoost", "GLOVE.6B.100d"] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "XGBoost", FIGURES+"xgboost", class_mapping=class_mapping)
    
    
    print("\n\nLog: GLOVE.6B.100d Extraction Complete\n\n")