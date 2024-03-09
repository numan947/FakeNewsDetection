import os
import sys
sys.path.append('./')

import pandas as pd
import numpy as np
import DataSetReader
import FeatureExtractor
import Models
import Preprocessor
import Visualization
from sklearn.metrics import confusion_matrix

##INFO: Primary columns: statement, stmt, label | stmt is the preprocessed statement

CUSTOM_STOP_WORDS = ["said", "say", "says", "tell", "tells", "told", "ask", "asks", "asked"]
y_str = ['barely-true', 'false', 'half-true', 'mostly-true', 'pants-fire', 'true']
# Map the string labels to numerical values
class_mapping = {
    'pants-fire': 0,
    'false': 1,
    'barely-true': 2,
    'half-true': 3,
    'mostly-true': 4,
    'true': 5
}


Visualization.FIG_DIR = "../generated/figures/liar_data/"
if os.path.exists(Visualization.FIG_DIR) == False:
    os.makedirs(Visualization.FIG_DIR)

############### VISUALIZATION ####################
def create_word_cloud(train_data, valid_data, test_data):
    Visualization.word_clouds_for_liar_data(train_data, "liar_data_train")
    Visualization.word_clouds_for_liar_data(valid_data, "liar_data_valid")
    Visualization.word_clouds_for_liar_data(test_data, "liar_data_test")
    
def visualize_data(train_data, valid_data, test_data):
     # create word clouds
    create_word_cloud(train_data, valid_data, test_data)
    Visualization.create_count_plot(train_data, "label", "Liar-Dataset: Train Data Label Distribution", "Label", "Count", "train_label_count")
    Visualization.create_count_plot(valid_data, "label", "Liar-Dataset: Validation Data Label Distribution", "Label", "Count", "valid_label_count")
    Visualization.create_count_plot(test_data, "label", "Liar-Dataset: Test Data Label Distribution", "Label", "Count", "test_label_count")



############### PREPROCESSING ####################
def preprocess_data(train_data, valid_data, test_data):
    # Preprocess the data
    train_data['stmt'] = Preprocessor.preprocess_text(train_data['statement'], CUSTOM_STOP_WORDS)
    valid_data['stmt'] = Preprocessor.preprocess_text(valid_data['statement'], CUSTOM_STOP_WORDS)
    test_data['stmt'] = Preprocessor.preprocess_text(test_data['statement'], CUSTOM_STOP_WORDS)
    
    return train_data, valid_data, test_data


############### FEATURE EXTRACTION ####################
def extract_cbow(train_data, valid_data, test_data):
    ## Bag of Words
    count_vectors_train, count_vectorizer = FeatureExtractor.get_bag_of_words(train_data['stmt'], max_features=15000) #HYPERPARAMETER
    count_vectors_valid = count_vectorizer.transform(valid_data['stmt'])
    count_vectors_test = count_vectorizer.transform(test_data['stmt'])
    
    X_train = count_vectors_train
    y_train = train_data['label']
    X_valid = count_vectors_valid
    y_valid = valid_data['label']
    X_test = count_vectors_test
    y_test = test_data['label']

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def extract_tfidf(train_data, valid_data, test_data):
    ## Bag of Words
    tfidf_vectors_train, tfidf_vectorizer = FeatureExtractor.get_tfidf(train_data['stmt'], max_features=15000) #HYPERPARAMETER
    tfidf_vectors_valid = tfidf_vectorizer.transform(valid_data['stmt'])
    tfidf_vectors_test = tfidf_vectorizer.transform(test_data['stmt'])
    
    X_train = tfidf_vectors_train
    y_train = train_data['label']
    X_valid = tfidf_vectors_valid
    y_valid = valid_data['label']
    X_test = tfidf_vectors_test
    y_test = test_data['label']

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, title, filename, class_mapping=None):
    pred_y_train = model.predict(X_train)
    pred_y_valid = model.predict(X_valid)
    pred_y_test = model.predict(X_test)
    if class_mapping:
        y_train = y_train.map(class_mapping)
        y_valid = y_valid.map(class_mapping)
        y_test = y_test.map(class_mapping)
    
    cm_train = confusion_matrix(y_train, pred_y_train)
    cm_valid = confusion_matrix(y_valid, pred_y_valid)
    cm_test = confusion_matrix(y_test, pred_y_test)
    
    reversed_class_mapping = {v: k for k, v in class_mapping.items()} if class_mapping else None
    classes = model.classes_ if class_mapping == None else [reversed_class_mapping[i] for i in model.classes_]
    
    Visualization.create_confusion_matrix(
        cm_train, 
        classes,
        title+" - Train Data", 
        filename+"_train.pdf"
    )
    
    Visualization.create_confusion_matrix(
        cm_valid, 
        classes, 
        title+" - Validation Data", 
        filename+"_valid.pdf"
    )
    
    Visualization.create_confusion_matrix(
        cm_test, 
        classes, 
        title+" - Test Data", 
        filename+"_test.pdf"
    )


def liar_multi_class():
    # read the data
    print("Log: Reading Liar Data")
    train_data, valid_data, test_data = DataSetReader.read_liar_data()
    
    # preprocess the data
    print("Log: Preprocessing Liar Data")
    train_data, valid_data, test_data = preprocess_data(train_data, valid_data, test_data)
    
    # visualize the data
    # visualize_data(train_data, valid_data, test_data)
    
    
    # Feature Extraction
    # 1. Bag of Words
    # 2. TF-IDF
    # 3. Word Embeddings
    # 4. BERT Embeddings
    # 5. GloVe Embeddings
    # 6. FastText Embeddings
    # 7. Word2Vec Embeddings
    # 8. Doc2Vec Embeddings
    # 9. LLM Embeddings
    
    metrics_df = pd.DataFrame(columns=[
        "Model", "Feature_Extraction_Method",
        "Train_Accuracy", "Validation_Accuracy", "Test_Accuracy", 
        "Train_F1", "Validation_F1", "Test_F1", 
        "Train_Precision", "Validation_Precision", "Test_Precision", 
        "Train_Recall", "Validation_Recall", "Test_Recall", 
        "Train_AUC", "Validation_AUC", "Test_AUC"
        ]
    )
    
    metrics_data = []
    
    
    # 1. Bag of Words
    print("\n\nLog: Extracting Bag of Words")
    CBOW_FIGURES = "../generated/figures/liar_data/cbow/"
    if os.path.exists(CBOW_FIGURES) == False:
        os.makedirs(CBOW_FIGURES)
        
    X_train, y_train, X_valid, y_valid, X_test, y_test = extract_cbow(train_data, valid_data, test_data)


    model, metrics = Models.fit_logistic_regression(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["Logistic Regression", "Bag of Words"] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "Logistic Regression", CBOW_FIGURES+"logistic_regression")
    
    model, metrics = Models.fit_decision_tree(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["Decision Tree", "Bag of Words"] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "Decision Tree", CBOW_FIGURES+"decision_tree")
    
    model, metrics = Models.fit_random_forest(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["Random Forest", "Bag of Words"] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "Random Forest", CBOW_FIGURES+"random_forest")
    
    model, metrics = Models.fit_svm(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["SVM", "Bag of Words"] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "SVM", CBOW_FIGURES+"svm")
    
    model, metrics = Models.fit_knn(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["KNN", "Bag of Words"] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "KNN", CBOW_FIGURES+"knn")
    
    model, metrics = Models.fit_gaussian_nb(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["Gaussian NB", "Bag of Words"] + metrics)
    create_confusion_matrix(model, X_train.toarray(), y_train, X_valid.toarray(), y_valid, X_test.toarray(), y_test, "Gaussian NB", CBOW_FIGURES+"gaussian_nb")
    
    model, metrics = Models.fit_adaboost(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["AdaBoost", "Bag of Words"] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "AdaBoost", CBOW_FIGURES+"adaboost")
    
    model, metrics = Models.fit_extra_trees(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["Extra Trees", "Bag of Words"] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "Extra Trees", CBOW_FIGURES+"extra_trees")
    
    model, metrics = Models.fit_xgboost(X_train, y_train, X_valid, y_valid, X_test, y_test, class_mapping=class_mapping)
    metrics_data.append(["XGBoost", "Bag of Words"] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "XGBoost", CBOW_FIGURES+"xgboost", class_mapping=class_mapping)
    
    
    print("\n\nLog: Bag of Words Extraction Complete\n\n")
    
    
    # 2. TF-IDF
    print("\n\nLog: Extracting TF-IDF")
    TFIDF_FIGURES = "../generated/figures/liar_data/tfidf/"
    if os.path.exists(TFIDF_FIGURES) == False:
        os.makedirs(TFIDF_FIGURES)
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = extract_tfidf(train_data, valid_data, test_data)
   
   
    model, metrics = Models.fit_logistic_regression(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["Logistic Regression", "TF-IDF"] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "Logistic Regression", TFIDF_FIGURES+"logistic_regression")
    
    model, metrics = Models.fit_decision_tree(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["Decision Tree", "TF-IDF"] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "Decision Tree", TFIDF_FIGURES+"decision_tree")
    
    model, metrics = Models.fit_random_forest(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["Random Forest", "TF-IDF"] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "Random Forest", TFIDF_FIGURES+"random_forest")
    
    model, metrics = Models.fit_svm(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["SVM", "TF-IDF"] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "SVM", TFIDF_FIGURES+"svm")
    
    model, metrics = Models.fit_knn(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["KNN", "TF-IDF"] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "KNN", TFIDF_FIGURES+"knn")
    
    model, metrics = Models.fit_gaussian_nb(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["Gaussian NB", "TF-IDF"] + metrics)
    create_confusion_matrix(model, X_train.toarray(), y_train, X_valid.toarray(), y_valid, X_test.toarray(), y_test, "Gaussian NB", TFIDF_FIGURES+"gaussian_nb")
    
    model, metrics = Models.fit_adaboost(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["AdaBoost", "TF-IDF"] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "AdaBoost", TFIDF_FIGURES+"adaboost")
    
    model, metrics = Models.fit_extra_trees(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["Extra Trees", "TF-IDF"] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "Extra Trees", TFIDF_FIGURES+"extra_trees")
    
    model, metrics = Models.fit_xgboost(X_train, y_train, X_valid, y_valid, X_test, y_test, class_mapping)
    metrics_data.append(["XGBoost", "TF-IDF"] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "XGBoost", TFIDF_FIGURES+"xgboost", class_mapping=class_mapping)
    
    
    print("\n\nLog: TF-IDF Extraction Complete\n\n")
    
    
    
    for row in metrics_data:
        metrics_df.loc[len(metrics_df)] = row
    
    metrics_df.to_csv("../generated/liar_data_metrics_multi_class.csv", index=False)
    




if __name__ == "__main__":
    liar_multi_class()