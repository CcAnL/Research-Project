import jsonlines
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tag import pos_tag
from sklearn.tree import DecisionTreeClassifier
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk.tokenize
from sklearn.linear_model import LinearRegression
from nltk.corpus import stopwords
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
import math
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from collections import Counter
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
#load the json file
#get the user_reviews and scores
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
                    user_scores.append(float(review['score']))

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
        tokenized = nltk.word_tokenize(text[0])
        stopwords_filtered = ""
#pos_tag is used to lemmatization
        for word in tokenized:
#get the tag of every word in one review
            a=pos_tag([word])[0][1]
            if a.startswith("VB"):
                newWord = lemmatizer.lemmatize(word, pos="v")
            elif a.startswith("NN"):
                newWord = lemmatizer.lemmatize(word, pos="n")
            elif a.startswith("JJ"):
                newWord = lemmatizer.lemmatize(word, pos="a")
            elif a.startswith("R"):
                newWord = lemmatizer.lemmatize(word, pos="r")
            else:
                newWord = lemmatizer.lemmatize(word)
            if newWord not in stopwords:
              stopwords_filtered = stopwords_filtered+" "+newWord
        preprocessed_texts.append(stopwords_filtered)
    return preprocessed_texts
processed_train_text=preprocess_events(preprocess_text)

#split the data into train dev test (80% 10% 10%)
train_set,rest_set, train_classes, rest_classes=train_test_split(processed_train_text,
                                                                 user_scores,
                                                                 test_size=0.2,
                                                                 )
dev_set,test_set, dev_classes, test_classes=train_test_split(rest_set,
                                                            rest_classes,
                                                            test_size=0.5,
                                                                 )
#transform the data according to tfidf
vectorizer = TfidfVectorizer()
train=vectorizer.fit_transform(train_set)
dev=vectorizer.transform(dev_set)
test=vectorizer.transform(test_set)

#train the models.
#evaluation metrics and visualization
def algorithms(clfs):
  for i, clf in enumerate(clfs):
    clf.fit(train,train_classes)
    y_pred_train = clf.predict(test)
    RMSE = math.sqrt(mean_squared_error(test_classes, y_pred_train))
    MAE=mean_absolute_error(test_classes, y_pred_train)
    print(clf)
    print(RMSE)
    print(MAE)
    plt.figure()
    plt.ylabel("scores")
    plt.title(clf)
    plt.plot(y_pred_train, 'r', label='predicted scores')
    plt.plot(test_classes, 'b', label='true scores')
    plt.legend()
    plt.savefig('./test_{}.jpg'.format(i))

linear = LinearRegression()
ridge = Ridge()
lasso = Lasso()
elasticnet = ElasticNet()
clfs=[]
clfs.append(linear)
clfs.append(lasso)
clfs.append(elasticnet)
clfs.append(ridge)
algorithms(clfs)
