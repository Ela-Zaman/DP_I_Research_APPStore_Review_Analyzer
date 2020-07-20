
import pymysql
import numpy as np


from nltk.corpus import wordnet
import heapq


import pandas as pd
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score



import nltk

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

class Features_or_UX:

    def __init__(self):
        self.m_s = {}

    def tokenize(self,msg):

        tokens = nltk.word_tokenize(msg)
        # remove punctuation and convert lower
        token_words = [w.lower() for w in tokens if w.isalpha()]
        return token_words

    def POST_tagger(self,msg):
        tagged = nltk.pos_tag(msg)
        review = []
        r = []
        i = 0
        # extract noun,verb,adjective from the reviews
        for _ in tagged:
            if _[1].startswith('VB') or _[1].startswith('J')  or _[1].startswith('NN'):
                r.append(_[0])

        return r

    # remove stopwords
    def remove_stopwords(self,wordslist):
        stop_words = list(stopwords.words('english'))
        meaningful_words = []

        custom = ['i', 'me', 'up', 'my', 'myself', 'we', 'our', 'ours',
                  'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
                  'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
                  'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'us',
                  'themselves', 'am', 'is', 'are', 'a', 'an', 'the', 'and', 'in',
                  'out', 'on', 'up', 'down', 's', 't', 'please', 'fix', 'app', 'fine', "application",
                  "app", "great", "enjoy", "game", "brawler","good","bad","love","nothing"]
        for w in wordslist:

            if w not in stop_words and w not in custom:
                meaningful_words.append(w)

        return meaningful_words


    def get_wordnet_pos(self,word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB
                    }

        return tag_dict.get(tag, wordnet.NOUN)

    def Lemmetizer(self,wordslist):
        # convert into lowercase

        lemmatizer = WordNetLemmatizer()

        lemmetized_list = [lemmatizer.lemmatize(w, self.get_wordnet_pos(w)) for w in wordslist]
        return (lemmetized_list)

    # data_cleaning method
    def data_cleaning(self,df):

        words_list = [self.tokenize(msg) for msg in df]

      #  print(words_list)

        Pos = [self.POST_tagger(word) for word in words_list]
        #print(Pos)
        clean = [self.remove_stopwords(p) for p in Pos]
        #print(clean)


        # mean_word=[remove_stopwords(w) for w in words_list]
        # print('mean_word: ', mean_word,end='\n')

        lemmetized_word = [self.Lemmetizer(w) for w in clean if len(w)>3]
        # print("l", lemmetized_word[1])

        r=[x for x in lemmetized_word if x]
        syn_extract = [self.merge_synonyms(p) for p in r]

        return syn_extract

    def unique_word_frequency(self,g_words_list):
        unique = []
        # count the unique words in the full dataset
        word_frequency = {}
        for words in g_words_list:
            for w in words:
                if w not in word_frequency.keys():
                    word_frequency[w] = 1
                else:
                    word_frequency[w] += 1

        return word_frequency




    def most_freq_words(self,unique):
        most_freq = heapq.nlargest(100, unique, key=unique.get)
        return most_freq

    def find_ngrams(self,input_list, n):
        return zip(*[input_list[i:] for i in range(n)])

    def extract_bigram_words(self,input_data):
        return self.find_ngrams(input_data, 2)

    def collocation(self,r):
        feature=[]
        for i in r:
            bigrams =self.extract_bigram_words(i)
            bigram_features = dict()
            for (w1, w2) in bigrams:
                bigram_features[(w1, w2)] = 1
            l = []

            for key in bigram_features.keys():
                l.append(key)
            feature.append(l)


        word_frequency = {}
        for words in feature:
            for w in words:
                if w not in word_frequency.keys():
                    word_frequency[w] = 1


                else:
                    word_frequency[w] += 1
        f_list = []

        for k in word_frequency.keys():
            if word_frequency[k] > 2:

                f_list.append([list(k), word_frequency[k]])
        t=sorted(f_list)
        feature_lists=[]

        for i in t:
            feature_lists.append(i[0][0]+'_'+i[0][1])




        return(feature_lists)






    # word_synonyms = get_word_synonyms_from_sent(word, sent)
    # print ("WORD:", word)
    # print ("SENTENCE:", sent)

    def syn_list(self, r):
        syn = []
        for synset in wordnet.synsets(r):
            for lemma in synset.lemmas():
                syn.append(lemma.name())
        return syn

    def merge_synonyms(self, k):
        for i in range(len(k)):
            if k[i] not in self.m_s.keys():
                for f, g in self.m_s.items():
                    if k[i] in g:
                        k[i] = f
                    else:
                        k[i]=k[i]
                self.m_s[k[i]] = self.syn_list(k[i])
        return k

    def get_data_features(self):

        connection = pymysql.connect(host='localhost',
                                     user='root',
                                     db='research')
        cursor = connection.cursor()
        sql = 'SELECT reviewId,title,comment FROM `feature_or_improvment_request_data'
        cursor.execute(sql)
        result = cursor.fetchall()
        f = pd.DataFrame(result, columns=['r_id', 'title', 'comment'])

        connection.close()
        return f
    def get_data_classification(self):
        f = self.get_data_features()
        f = f.assign(label='features')
        connection = pymysql.connect(host='localhost',
                                     user='root',
                                     db='research')
        cursor = connection.cursor()
        sql = 'SELECT reviewId,title,comment FROM `not_feature_or_improvment_request_data`'

        cursor.execute(sql)
        result = cursor.fetchall()
        n_f = pd.DataFrame(result, columns=['r_id', 'title', 'comment'])
        n_f = n_f.assign(label='not_features')
        data= pd.concat([f, n_f])
        data = data.sample(frac=1).reset_index(drop=True)
        #print(data)





        connection.close()
        return data

    def feature_vector(self,dataset):
        wordfreq = self.unique_word_frequency(dataset)
        most_freq = self.most_freq_words(wordfreq)
        features = []
        for wordslist in dataset:
            feature_vector = {}
            for m in most_freq:
                # count the frequency of a word in each message
                feature_vector[m] = wordslist.count(m)

            features.append(feature_vector)
        # create a dataframe of features
        df = pd.DataFrame(features)
        return df

    def feature_or_User_experience_classification(self,clf):
        lb = LabelBinarizer()
        # Multinomial Naive Bayes

        df=self.get_data_classification()
        #print(df)
        # feature
        X = self.feature_vector(self.data_cleaning(df['comment']))
        #print(X)
        # class
        y = df['label']
        # Initialize the accuracy of the models
        recall = []
        precision = []
        accuracy = []
        # Task7
        # Split the feature_vector using 10 fold cross validation
        kf = KFold(n_splits=10, shuffle=False)

        # Iterate over each train-test split
        for train_index, test_index in kf.split(X):
            # Split train-test
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # Task 8

            # Multinomial Naive Bayes

            model_n_b = clf.fit(X_train, y_train)
            # prediction
            test = model_n_b.predict(X_test)


            y_train = np.array([number[0] for number in lb.fit_transform(y_train)])

            accuracy = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
            recall = cross_val_score(clf, X_train, y_train, cv=10, scoring='recall')
            precision = cross_val_score(clf, X_train, y_train, cv=10, scoring='precision')







        result=[['Accuracy: ',np.mean(accuracy)],['Precision',np.mean(precision)],['Recall: ',np.mean(recall)]]

        return result

    def get_accuracy(self):
        nb=MultinomialNB()
        NB=self.feature_or_User_experience_classification(nb)
        print('Multinomial Naive Bayes Classifier: ',NB)
        dt = DecisionTreeClassifier()
        DT = self.feature_or_User_experience_classification(dt)
        print('Decision Tree Classifier: ', DT)






    def get_features(self):
        f=self.get_data_features()
        f['merged']=f['title'].astype(str) + f['comment']
        f= f.assign(label='features')

        processed=self.data_cleaning(f['merged'])
        #un=self.unique_word_frequency(processed)
        #m_w=self.most_freq_words(un)
        #print(m_w)
       # r=self.get_data_classification()

        extract_features=self.collocation(processed)

        return extract_features




f=Features_or_UX()
print(f.get_features())
f.get_accuracy()
