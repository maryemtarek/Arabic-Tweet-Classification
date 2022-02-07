import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import arabicstopwords.arabicstopwords as stp
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
import pyarabic.araby as araby




negative = pd.read_csv(r'\train_Arabic_tweets_negative_20190413.tsv', sep='\t')
positive = pd.read_csv(r'\train_Arabic_tweets_positive_20190413.tsv', sep='\t')

all_data = [negative[:1000],positive[:1000]]
for df in all_data:
     df.columns = ['label', 'tweet']
data = pd.concat(all_data).reset_index(drop=True)
data = data.sample(frac=1) #shuffeling data

#for index,tweet in enumerate(data['tweet']):
   # data['tweet'][index] = araby.strip_tashkeel(tweet)
   # data['tweet'][index] = araby.strip_tatweel(tweet)
   

data['tweet'] = data['tweet'].str.replace('\d+','') #removing numbers
data["tweet"] = data["tweet"].str.replace('[^\w\s]','') #removing punctuation and emojis
data['tweet'] = data['tweet'].str.replace('_',' ') #removing underscore '_'
data['tweet'] = data['tweet'].str.replace('  ',' ') #removing the double spaces
data['tweet'] = data['tweet'].str.replace('[a-zA-Z]','') #removing english chars




 #removing hamazat   
data['tweet'] = data['tweet'].str.replace("آ","ا") 
data['tweet'] = data['tweet'].str.replace("أ","ا")
data['tweet'] = data['tweet'].str.replace("إ","ا")

freqs = {}
features = []
feature_list = []

#tokenizing the tweets
tweet_tokens = pd.DataFrame(columns=['token','label'])
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
for ind in data.index:
    tweet_tokens = tweet_tokens.append({'token':tokenizer.tokenize(data['tweet'][ind]),'label':data['label'][ind]},ignore_index=True)
    
#removing stop words
for index,tweet in enumerate(tweet_tokens['token']):
    for word in tweet:
        if stp.is_stop(word):
            tweet_tokens['token'][index] = tweet
            tweet.remove(word)
    tweet_tokens['token'][index] = tweet

#replacing the labels with 0,1
tweet_tokens['label'].replace({'pos':1,'neg':0},inplace=True)

#calculating words freqs
for index in range(len(tweet_tokens)):
    
    token = tweet_tokens['token'][index]
    label = tweet_tokens['label'][index]            
    for word in token:
        pair = (word, label)
        if pair in freqs:
            freqs[pair] += 1
        else:
            freqs[pair] = 1    

#feature extraction
for index in range(len(tweet_tokens)):
    positive_count = 0
    negative_count = 0
    tokens = tweet_tokens['token'][index]
    label = tweet_tokens['label'][index]
    for token in tokens:
        positive_count += freqs.get((token,1),0)
        negative_count += freqs.get((token,0),0)
    features.append([1,positive_count,negative_count,label])
df = pd.DataFrame(features)

#spliting data into train , test using Kfold
kf = KFold(n_splits=20,shuffle=True,random_state=None)
X = df.iloc[ : , :3]
y = df[3]

accLog=[]
f1Log=[]
accsvm=[]
f1svm=[]
acctree=[]
f1tree=[]
accgnb=[]
f1gnb=[]

logistic_model = LogisticRegression()
svm_model = svm.SVC()
tree_model = DecisionTreeClassifier()
gnb = GaussianNB()
for train_index,test_index in kf.split(X,y):
    X_train = X.iloc[train_index, :3]
    X_test = X.iloc[test_index, :3]
    y_test =  y[test_index]
    y_train = y[train_index]
    
    
    logistic_model.fit(X_train,y_train)
    logistic_predict = logistic_model.predict(X_test)
    logistic_predict = np.expand_dims(logistic_predict,axis=1)
    logistic_score = metrics.accuracy_score(y_test,logistic_predict)
    accLog.append(logistic_score)
    logistic_fscore = f1_score(y_test, logistic_predict,average='macro')
    f1Log.append(logistic_fscore)
    
    
    svm_model.fit(X_train,y_train)
    svm_predict = svm_model.predict(X_test)
    svm_predict = np.expand_dims(svm_predict,axis=1)
    svm_score = metrics.accuracy_score(y_test,svm_predict)
    accsvm.append(svm_score)
    svm_fscore = f1_score(y_test, svm_predict,average='macro')
    f1svm.append(svm_fscore)
    
    
    tree_model.fit(X_train,y_train)
    tree_predict = tree_model.predict(X_test)
    tree_predict = np.expand_dims(tree_predict,axis=1)
    tree_score = metrics.accuracy_score(y_test,tree_predict)
    acctree.append(tree_score)
    tree_fscore = f1_score(y_test, tree_predict,average='macro')
    f1tree.append(tree_fscore)
    
    gnb.fit(X_train,y_train)
    gnb_predict = tree_model.predict(X_test)
    gnb_predict = np.expand_dims(gnb_predict,axis=1)
    gnb_score = metrics.accuracy_score(y_test,gnb_predict)
    accgnb.append(gnb_score)
    gnb_fscore = f1_score(y_test, gnb_predict,average='macro')
    f1gnb.append(gnb_fscore)

#y_test = np.expand_dims(y_test,axis=1)


avgacclog=sum(accLog)/len(accLog)
avgf1log=sum(f1Log)/len(f1Log)
print("logistic score:"+str(avgacclog))
print("logistic f-score:"+str(avgf1log)+'\n')

avgaccsvm=sum(accsvm)/len(accsvm)
avgf1svm=sum(f1svm)/len(f1svm)
print("SVM score:"+str(avgaccsvm))
print("SVM f-score:"+str(avgf1svm)+'\n')

avgacctree=sum(acctree)/len(acctree)
avgf1tree=sum(f1tree)/len(f1tree)
print("tree score:"+str(avgacctree))
print("tree f-score:"+str(avgf1tree)+'\n')


avgaccgnb=sum(accgnb)/len(accgnb)
avgf1gnb=sum(f1gnb)/len(f1gnb)
print("\ngnb score "+str(avgaccgnb))
print("gnb f-score "+str(avgf1gnb)+'\n')
