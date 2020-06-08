import jsonlines
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.tree import DecisionTreeClassifier
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk.tokenize
from nltk.corpus import stopwords
from sklearn.neural_network import MLPClassifier
from collections import Counter
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
#load the json file
def get_json():
    with open("game_spider_20_06_2015.jsonlines") as json_file:
     content=[]
     user_scores=[]
     for item in jsonlines.Reader(json_file):
            user_reviews=item['user_reviews']
            for review in user_reviews:
                if review['text'] != "" :
                    text2=[]
                    text2.append(review['text'])
                    content.append(text2)
                    user_scores.append(review['score'])

     return content,user_scores


preprocess_text,user_scores=get_json()


#create a bow
#tokenize
#remove stopword

stopwords = set(stopwords.words('english'))

def preprocess_events(texts):
    preprocessed_texts = []
    lemmatizer = WordNetLemmatizer()

    for text in texts:
        tokenized = nltk.word_tokenize(text)
        stopwords_filtered = ""

        for word in tokenized:
            a=word.lower()
            if a.startswith("VB"):
                newWord = lemmatizer.lemmatize(a, pos="v")
            elif a.startswith("NN"):
                newWord = lemmatizer.lemmatize(a, pos="n")
            elif a.startswith("JJ"):
                newWord = lemmatizer.lemmatize(a, pos="a")
            elif a.startswith("R"):
                newWord = lemmatizer.lemmatize(a, pos="r")
            else:
                newWord = lemmatizer.lemmatize(a)
            if newWord not in stopwords:
              stopwords_filtered = stopwords_filtered+" "+newWord
        preprocessed_texts.append(stopwords_filtered)
    return preprocessed_texts
processed_train_text=preprocess_events(preprocess_text)

#processed_dev_text=preprocess_events(dev_text)

#vectorizer = TfidfVectorizer(ngram_range=(2,2))
#train_set=vectorizer.fit_transform(processed_train_text)
#dev_set=vectorizer.transform(processed_dev_text)

