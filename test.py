import json
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer

import nltk.tokenize
from nltk.corpus import stopwords
from collections import Counter
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
#load the json file
def get_json(file_path):
    with open(file_path) as json_file:
      data=json.load(json_file)
      content=[]
      for key in data:
          text=[]
          text.append(data[key]['text'])
          content.append(text)
      return content
#preprocess_text=get_json("as.json")


dev_text=get_json("dev.json")
print(dev_text)
#dev_text=get_json("test-unlabelled.json")

#create a bow
#tokenize
#remove stopword

stopwords = set(stopwords.words('english'))

def preprocess_events(texts):
    preprocessed_texts = []
    lemmatizer = WordNetLemmatizer()
   # stopwords.add('climate')
   # stopwords.add('change')
    #stopwords.add('CNN')
    #stopwords.add('Written')
    # stopwords.add(',')
    # stopwords.add('.')
    # stopwords.add('’')
    # stopwords.add('``')
    # stopwords.add(':')
    # stopwords.add('@')
    # stopwords.add('?')
    # stopwords.add('“')
    for text in texts:
        tokenized = nltk.word_tokenize(text[0])
        stopwords_filtered = []

        for word in tokenized:
            if word.startswith("VB"):
                newWord = lemmatizer.lemmatize(word, pos="v")
            elif word.startswith("NN"):
                newWord = lemmatizer.lemmatize(word, pos="n")
            elif word.startswith("JJ"):
                newWord = lemmatizer.lemmatize(word, pos="a")
            elif word.startswith("R"):
                newWord = lemmatizer.lemmatize(word, pos="r")
            else:
                newWord = lemmatizer.lemmatize(word)
            if newWord not in stopwords :
                stopwords_filtered.append(newWord)
        preprocessed_texts.append(dict(Counter(stopwords_filtered)))
    return preprocessed_texts
processed_train_text=preprocess_events(preprocess_text)
processed_dev_text=preprocess_events(dev_text)
print(processed_train_text)
vectorizer = DictVectorizer(ngram_range=(2,2))
train_set=vectorizer.fit_transform(processed_train_text)
dev_set=vectorizer.transform(processed_dev_text)
def byes(train_data,dev_data):
    clf=svm.SVC()
    #clf = MultinomialNB(alpha=0.05)
    #clf=LogisticRegression(C=1)
    train_class=[]
    for i in range(1168): train_class.append(1)
    for j in range(3151): train_class.append(0)
    clf.fit(train_data,train_class)
    y_pred_train = clf.predict(dev_data)
    print(y_pred_train )
    return getResult(y_pred_train)

def getResult(predict):
    i=0
  #  f=open("test-unlabelled.json")
    f = open("dev.json")
    data = json.load(f)
    for key in data:
        del data[key]['text']
        data[key]['label']=str(predict[i])
        i+=1
    return data
#result=oneClassSvm(train_set,dev_set)
# def oneClassSvm(train_data,dev_data):
#     clf = svm.OneClassSVM(nu=0.35, kernel='rbf')
#     clf.fit(train_data)
#     y_pred_train = clf.predict(dev_data)
#     return getResult(y_pred_train)
result= byes(train_set,dev_set)
json_str = json.dumps(result)
with open('test15.json', 'w') as json_file:
    json_file.write(json_str)