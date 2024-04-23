from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


## should only be used for training data
## for transforming test data, use the transform method
## of the vectorizer object returned by this function
def get_bag_of_words(text_data, max_features=1000):
	vectorizer = CountVectorizer(max_features=max_features)
	bag_of_words = vectorizer.fit_transform(text_data)
	return bag_of_words, vectorizer


## should only be used for training data
## for transforming test data, use the transform method
## of the vectorizer object returned by this function
def get_tfidf(text_data, max_features=1000):
	vectorizer = TfidfVectorizer(max_features=max_features, )
	tfidf = vectorizer.fit_transform(text_data)
	return tfidf, vectorizer


if __name__ == "__main__":
	text_data = ["The quick brown fox jumped over the lazy dog.",
				 "The dog.",
				 "The fox"]
	bag_of_words, vectorizer = get_bag_of_words(text_data)
	print(bag_of_words.toarray())
	print(vectorizer)

	tfidf, vectorizer = get_tfidf(text_data)
	print(tfidf.toarray())
	print(vectorizer)