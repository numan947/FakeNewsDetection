import os
import sys

from analysis import *
sys.path.append('./')
from tqdm import tqdm 
import pandas as pd
import numpy as np
import DataSetReader
import FeatureExtractor
import Models
import Preprocessor
import Visualization
from sklearn.metrics import confusion_matrix

##INFO: Primary columns: statement, stmt, label | stmt is the preprocessed statement

CUSTOM_STOP_WORDS = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", 
             "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during",
             "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", 
             "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
             "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or",
             "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", 
             "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
             "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
             "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's",
             "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've",
             "your", "yours", "yourself", "yourselves" ]
             
             
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
# class_mapping = {
#     'pants-fire': 0,
#     'false': 0,
#     'barely-true': 0,
#     'half-true': 1,
#     'mostly-true': 1,
#     'true': 1
# }


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
    do_cbow(train_data, valid_data, test_data, metrics_data, smote=True)
    do_tfidf(train_data, valid_data, test_data, metrics_data, smote=True)
    do_glove_6b(train_data, valid_data, test_data, metrics_data, embedding_name="glove.6B.300d", agg='mean', smote=True)
    do_glove_6b(train_data, valid_data, test_data, metrics_data, embedding_name="glove.6B.300d", agg='max', smote=True)
    do_glove_6b(train_data, valid_data, test_data, metrics_data, embedding_name="glove.6B.300d", agg='sum', smote=True)
    
    do_glove_6b(train_data, valid_data, test_data, metrics_data, embedding_name="glove.twitter.27B.200d", agg='mean', smote=True)
    do_glove_6b(train_data, valid_data, test_data, metrics_data, embedding_name="glove.twitter.27B.200d", agg='max', smote=True)
    do_glove_6b(train_data, valid_data, test_data, metrics_data, embedding_name="glove.twitter.27B.200d", agg='sum', smote=True)
    
    do_glove_6b(train_data, valid_data, test_data, metrics_data, embedding_name="glove.840B.300d", agg='mean', smote=True)
    do_glove_6b(train_data, valid_data, test_data, metrics_data, embedding_name="glove.840B.300d", agg='max', smote=True)
    do_glove_6b(train_data, valid_data, test_data, metrics_data, embedding_name="glove.840B.300d", agg='sum', smote=True)
    
    
    # do_fasttext(train_data, valid_data, test_data, metrics_data, embedding_name="crawl-300d-2M", agg='mean')
    
    
    for row in metrics_data:
        metrics_df.loc[len(metrics_df)] = row
    
    metrics_df.to_csv("../generated/liar_data_metrics_smote_6_class.csv", index=False)




if __name__ == "__main__":
    liar_multi_class()