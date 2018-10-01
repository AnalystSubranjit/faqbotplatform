import pandas as pd
import numpy as np
import pickle
import operator
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.metrics.pairwise import cosine_similarity
import random
import nltk
from nltk.stem.lancaster import LancasterStemmer


class BankFaqs:
    def __init__(self, faqslist):
        self.faqslist = faqslist
        self.stemmer = LancasterStemmer()
        self.le = LE()
        self.vectorizer = TfidfVectorizer(min_df=1, stop_words='english')   
        dataframeslist = [pd.read_csv(csvfile).dropna() for csvfile in self.faqslist]
        self.data = pd.concat(dataframeslist,  ignore_index=True)
        self.questions = self.data['Question'].values     
        
        self.build_model()
        
    def cleanup(self, sentence):
        word_tok = nltk.word_tokenize(sentence)
        stemmed_words = [self.stemmer.stem(w) for w in word_tok]
        return ' '.join(stemmed_words)
        
    def build_model(self):
        X = []
        for question in self.questions:
            X.append(self.cleanup(question))
        
        self.vectorizer.fit(X)
        self.le.fit(self.data['Class'])
        
        X = self.vectorizer.transform(X)
        y = self.le.transform(self.data['Class'])
        
        
        trainx, testx, trainy, testy = tts(X, y, test_size=.25, random_state=42)
        
        self.model = SVC(kernel='linear')
        self.model.fit(trainx, trainy)
        # print("SVC:", self.model.score(testx, testy))        
        
    def query(self, usr):
        #print("User typed : " + usr)
        try:
            t_usr = self.vectorizer.transform([self.cleanup(usr.strip().lower())])
            class_ = self.le.inverse_transform(self.model.predict(t_usr)[0])
            #print("Class " + class_)
            questionset = self.data[self.data['Class']==class_]
            
            #threshold = 0.7
            cos_sims = []
            for question in questionset['Question']:
                sims = cosine_similarity(self.vectorizer.transform([question]), t_usr)
                #if sims > threshold:
                cos_sims.append(sims)
                
            #print("scores " + str(cos_sims))                
            if len(cos_sims) > 0:
                ind = cos_sims.index(max(cos_sims)) 
                #print(ind)
                #print(questionset.index[ind])
                return self.data['Answer'][questionset.index[ind]]
        except Exception as e:
            print(e)
            return "Could not follow your question [" + usr + "], Try again"
    
    
