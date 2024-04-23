import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from LiarDriver import *
from tqdm import tqdm
import Models
import gc

def do_cbow(train_data, valid_data, test_data, metrics_data):
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

def do_tfidf(train_data, valid_data, test_data, metrics_data):

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


def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Loading GloVe embeddings", unit=" vectors"):
            values = line.split()
            word = values[0]
            embedding = np.array(values[1:], dtype='float32')
            embeddings_index[word] = embedding
    return embeddings_index

def load_840B_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Loading GloVe embeddings", unit=" vectors"):
            values = line.split()
            word = ''.join(values[:-300])
            vectors = np.asarray(values[-300:], dtype='float32')
            embeddings_index[word] = vectors
    return embeddings_index

def load_fasttext_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        n, d = map(int, f.readline().split())
        lines = f.readlines()
        for line in tqdm(lines, desc="Loading Fasttext embeddings", unit=" vectors"):
            tokens = line.rstrip().split(' ')
            embeddings_index[tokens[0]] = map(float, tokens[1:])
    return embeddings_index


# def load_fasttext_embeddings(file_path):
#     fin = io.open(file_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
#     n, d = map(int, fin.readline().split())
#     data = {}
#     for line in tqdm(fin, desc="Loading FastText embeddings", unit=" vectors"):
#         tokens = line.rstrip().split(' ')
#         data[tokens[0]] = map(float, tokens[1:])
#     return data

def generate_sentence_embedding(sentence, embeddings_index, aggregation_method='mean'):
    words = sentence.split()
    word_embeddings = [embeddings_index[word] for word in words if word in embeddings_index]
    if len(word_embeddings) == 0:
        return np.zeros_like(next(iter(embeddings_index.values())))
    else:
        if aggregation_method == 'mean':
            return np.mean(word_embeddings, axis=0)
        elif aggregation_method == 'sum':
            return np.sum(word_embeddings, axis=0)
        elif aggregation_method == 'max':
            return np.max(word_embeddings, axis=0)
        elif aggregation_method == 'min':
            return np.min(word_embeddings, axis=0)
        else:
            raise ValueError("Invalid aggregation method. Choose from 'mean', 'sum', 'max', or 'min'.")


def generate_embeddings_for_dataframe(df, embeddings_index, aggregation_method='mean', column='sentence'):
    row_indices = []
    col_indices = []
    data = []
    for i, sentence in enumerate(tqdm(df[column], desc="Generating embeddings", unit=" sentences")):
        sentence_embedding = generate_sentence_embedding(sentence, embeddings_index, aggregation_method)
        non_zero_indices = np.nonzero(sentence_embedding)[0]
        row_indices.extend([i] * len(non_zero_indices))
        col_indices.extend(non_zero_indices)
        data.extend(sentence_embedding[non_zero_indices])
    embeddings = csr_matrix((data, (row_indices, col_indices)), shape=(len(df), len(next(iter(embeddings_index.values())))))
    return embeddings

def extract_embeddings(train_data, valid_data, test_data, embeddings_index):
    X_train = generate_embeddings_for_dataframe(train_data, embeddings_index, column='stmt')
    y_train = train_data['label']
    X_valid = generate_embeddings_for_dataframe(valid_data, embeddings_index, column='stmt')
    y_valid = valid_data['label']
    X_test = generate_embeddings_for_dataframe(test_data, embeddings_index, column='stmt')
    y_test = test_data['label']
    return X_train, y_train, X_valid, y_valid, X_test, y_test

GLOVE_PATH = "../embeddings/glove/{}.txt"
def do_glove_6b(train_data, valid_data, test_data, metrics_data, embedding_name = 'glove.6B.100d', agg = 'mean'):

    # 2. GLOVE.6B.100d
    print("\n\nLog: Extracting {}".format(embedding_name))
    FIGURES = "../generated/figures/liar_data/{}/".format(embedding_name+"_"+agg)
    if os.path.exists(FIGURES) == False:
        os.makedirs(FIGURES)
    
    if '840B' in embedding_name:
        embeddings_index = load_840B_glove_embeddings(GLOVE_PATH.format(embedding_name))
    else:
        embeddings_index = load_glove_embeddings(GLOVE_PATH.format(embedding_name))
    X_train, y_train, X_valid, y_valid, X_test, y_test = extract_embeddings(train_data, valid_data, test_data, embeddings_index)
 
    
    model, metrics = Models.fit_logistic_regression(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["Logistic Regression", embedding_name+"_"+agg] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "Logistic Regression", FIGURES+"logistic_regression")
    gc.collect()
    
    model, metrics = Models.fit_decision_tree(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["Decision Tree", embedding_name+"_"+agg] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "Decision Tree", FIGURES+"decision_tree")
    gc.collect()
    
    model, metrics = Models.fit_random_forest(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["Random Forest", embedding_name+"_"+agg] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "Random Forest", FIGURES+"random_forest")
    gc.collect()
    
    model, metrics = Models.fit_svm(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["SVM", embedding_name+"_"+agg] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "SVM", FIGURES+"svm")
    gc.collect()
    
    
    model, metrics = Models.fit_knn(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["KNN", embedding_name+"_"+agg] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "KNN", FIGURES+"knn")
    gc.collect()
    
    
    model, metrics = Models.fit_gaussian_nb(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["Gaussian NB", embedding_name+"_"+agg] + metrics)
    create_confusion_matrix(model, X_train.toarray(), y_train, X_valid.toarray(), y_valid, X_test.toarray(), y_test, "Gaussian NB", FIGURES+"gaussian_nb")
    gc.collect()
    
    
    model, metrics = Models.fit_adaboost(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["AdaBoost", embedding_name+"_"+agg] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "AdaBoost", FIGURES+"adaboost")
    gc.collect()
    
    
    model, metrics = Models.fit_extra_trees(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["Extra Trees", embedding_name+"_"+agg] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "Extra Trees", FIGURES+"extra_trees")
    gc.collect()
    
    model, metrics = Models.fit_xgboost(X_train, y_train, X_valid, y_valid, X_test, y_test, class_mapping)
    metrics_data.append(["XGBoost", embedding_name+"_"+agg] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "XGBoost", FIGURES+"xgboost", class_mapping=class_mapping)
    gc.collect()
    
    
    
    print("\n\nLog: {} Extraction Complete\n\n".format(embedding_name+"_"+agg))


FASTTEXT_PATH = "../embeddings/fasttext/{}.vec"

def do_fasttext(train_data, valid_data, test_data, metrics_data, embedding_name = 'crawl-300d-2M', agg = 'mean'):

    # 2. crawl-300d-2M.vec
    print("\n\nLog: Extracting {}".format(embedding_name))
    FIGURES = "../generated/figures/liar_data/{}/".format(embedding_name+"_"+agg)
    if os.path.exists(FIGURES) == False:
        os.makedirs(FIGURES)
    
    embeddings_index = load_fasttext_embeddings(FASTTEXT_PATH.format(embedding_name))
    X_train, y_train, X_valid, y_valid, X_test, y_test = extract_embeddings(train_data, valid_data, test_data, embeddings_index)
 
    
    model, metrics = Models.fit_logistic_regression(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["Logistic Regression", embedding_name+"_"+agg] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "Logistic Regression", FIGURES+"logistic_regression")
    gc.collect()
    
    model, metrics = Models.fit_decision_tree(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["Decision Tree", embedding_name+"_"+agg] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "Decision Tree", FIGURES+"decision_tree")
    gc.collect()
    
    model, metrics = Models.fit_random_forest(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["Random Forest", embedding_name+"_"+agg] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "Random Forest", FIGURES+"random_forest")
    gc.collect()
    
    model, metrics = Models.fit_svm(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["SVM", embedding_name+"_"+agg] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "SVM", FIGURES+"svm")
    gc.collect()
    
    
    model, metrics = Models.fit_knn(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["KNN", embedding_name+"_"+agg] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "KNN", FIGURES+"knn")
    gc.collect()
    
    model, metrics = Models.fit_gaussian_nb(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["Gaussian NB", embedding_name+"_"+agg] + metrics)
    create_confusion_matrix(model, X_train.toarray(), y_train, X_valid.toarray(), y_valid, X_test.toarray(), y_test, "Gaussian NB", FIGURES+"gaussian_nb")
    gc.collect()
    
    model, metrics = Models.fit_adaboost(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["AdaBoost", embedding_name+"_"+agg] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "AdaBoost", FIGURES+"adaboost")
    gc.collect()
    
    model, metrics = Models.fit_extra_trees(X_train, y_train, X_valid, y_valid, X_test, y_test)
    metrics_data.append(["Extra Trees", embedding_name+"_"+agg] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "Extra Trees", FIGURES+"extra_trees")
    gc.collect()
    
    model, metrics = Models.fit_xgboost(X_train, y_train, X_valid, y_valid, X_test, y_test, class_mapping)
    metrics_data.append(["XGBoost", embedding_name+"_"+agg] + metrics)
    create_confusion_matrix(model, X_train, y_train, X_valid, y_valid, X_test, y_test, "XGBoost", FIGURES+"xgboost", class_mapping=class_mapping)
    gc.collect()
    
    print("\n\nLog: {} Extraction Complete\n\n".format(embedding_name+"_"+agg))