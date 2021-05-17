import jsonlines
import nltk
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
import nltk.tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

#get the content of the json file
def get_json():
    with open("game_spider_20_06_2015(2).jsonlines") as f:
        content = []
        user_scores = []
        critic_scores=[]
        critic_content=[]
        review_poaryity=[]
        num=0
        for item in jsonlines.Reader(f):#every item consists of critic review and user reviews of one game
            user_reviews = item['critic_reviews']
            num=num+1

            for review in user_reviews:#a review contains of data,source,score,text
                if review['text'] != "":
                    text2 = []
                    text2.append(review['text'])
                    content.append(text2)
                    # if float(review['score'])<=4:
                    #     polarity="negative"
                    # elif float(review['score'])>4 and float(review['score'])<6:
                    #     polarity = "neural"
                    # else:
                    #     polarity="positive"
                    # review_poaryity.append(polarity)
                    user_scores.append(float(review['score']))
        print(num)
        return content, user_scores
user_reviews,user_scores=get_json()
#print(user_reviews[0])
#print(len(user_scores))


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
            word=str.lower(word)
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

processed_train_usertext=preprocess_events(user_reviews)




train_set,test_set, train_classes, test_classes=train_test_split(processed_train_usertext,
                                                                 user_scores,
                                                                 test_size=0.2,
                                                                 )

test_polarity=[]
for i in test_classes:
    if i <= 40:
        test_polarity.append("negative")
    elif i > 40 and i < 60:
        test_polarity.append("neural")
    else:
        test_polarity.append("positive")

vectorizer = TfidfVectorizer()
train=vectorizer.fit_transform(train_set)
test=vectorizer.transform(test_set)


def algorithms(clfs):
    for clf in clfs:
        clf.fit(train,train_classes)
        pre_polarity=[]
        prediction=clf.predict(test)
        for i in prediction:
            if i <=40:pre_polarity.append("negative")
            elif i>40 and i<60: pre_polarity.append("neural")
            else: pre_polarity.append("positive")

        acc=accuracy_score(test_polarity,pre_polarity)
        print(acc)
        print(classification_report(test_polarity,pre_polarity))

linear = LinearRegression()
ridge = Ridge()
lasso = Lasso()
clfs=[]
clfs.append(linear)
clfs.append(lasso)
clfs.append(ridge)
algorithms(clfs)