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

##INFO: Primary columns: statement, stmt, label | stmt is the preprocessed statement

CUSTOM_STOP_WORDS = ["said", "say", "says", "tell", "tells", "told", "ask", "asks", "asked"]

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




def main():
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
    
    # 1. Bag of Words
    print("\n\nLog: Extracting Bag of Words")
    X_train, y_train, X_valid, y_valid, X_test, y_test = extract_cbow(train_data, valid_data, test_data)
    # Models.fit_logistic_regression(X_train, y_train, X_valid, y_valid, X_test, y_test)
    # Models.fit_decision_tree(X_train, y_train, X_valid, y_valid, X_test, y_test)
    # Models.fit_random_forest(X_train, y_train, X_valid, y_valid, X_test, y_test)
    # Models.fit_svm(X_train, y_train, X_valid, y_valid, X_test, y_test)
    # Models.fit_knn(X_train, y_train, X_valid, y_valid, X_test, y_test)
    Models.fit_gaussian_nb(X_train, y_train, X_valid, y_valid, X_test, y_test)
    print("\n\nLog: Bag of Words Extraction Complete\n\n")
    
    
    # 2. TF-IDF
    print("\n\nLog: Extracting TF-IDF")
    X_train, y_train, X_valid, y_valid, X_test, y_test = extract_tfidf(train_data, valid_data, test_data)
    # Models.fit_logistic_regression(X_train, y_train, X_valid, y_valid, X_test, y_test)
    # Models.fit_decision_tree(X_train, y_train, X_valid, y_valid, X_test, y_test)
    # Models.fit_random_forest(X_train, y_train, X_valid, y_valid, X_test, y_test)
    # Models.fit_svm(X_train, y_train, X_valid, y_valid, X_test, y_test)
    # Models.fit_knn(X_train, y_train, X_valid, y_valid, X_test, y_test)
    Models.fit_gaussian_nb(X_train, y_train, X_valid, y_valid, X_test, y_test)
    print("\n\nLog: TF-IDF Extraction Complete\n\n")
    
   
    




if __name__ == "__main__":
    main()