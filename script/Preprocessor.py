from tqdm import tqdm 
import re 
import nltk 
nltk.download('punkt') 
nltk.download('stopwords') 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem.porter import PorterStemmer 
import string


## This function preprocesses the text data by removing punctuations,
## converting to lower case, removing stop words and stemming the words.
## The function returns the preprocessed text data.
def preprocess_text(text_data, my_custom_stop_words=[]):
    all_stop= stopwords.words('english')
    all_stop.extend(my_custom_stop_words)
    porter = PorterStemmer()
    
    preprocessed_text = [] 
    for sentence in tqdm(text_data): 
        sentence = re.sub(r'[^\w\s]', '', sentence) # 
        sentence = sentence.lower()    
        tokens = word_tokenize(sentence)
        
        filtered_tokens = [word.translate(str.maketrans('', '', string.punctuation)) for word in tokens]
        
        filtered_tokens = [word for word in filtered_tokens if word]
        filtered_text = [porter.stem(word) for word in filtered_tokens if word not in all_stop]
        preprocessed_text.append(' '.join(filtered_text))
        
    return preprocessed_text


if __name__ == "__main__":
    print("Preprocessing Text Data")
    text_data = [
        "The quick brown fox jumps over the lazy dog", 
        "I am a sentence for which I would like to get its words"
        ]
    preprocessed_text = preprocess_text(text_data)
    print(preprocessed_text)
    print("Preprocessing Text Data Complete")