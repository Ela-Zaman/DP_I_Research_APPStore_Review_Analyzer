import pandas as pd
import pymysql
from nltk.collocations import *
from nltk.metrics.association import *
from nltk.corpus import brown


from nltk.corpus import wordnet
import heapq
import nltk

import pandas as pd

from nltk.collocations import *
import nltk
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import webtext
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

class get_Features:

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
                  'out', 'on', 'up', 'down', 's', 't', 'please', 'fix', 'app', "facebook", "fb", "lite", "application",
                  "app", "great", "enjoy", "game", "brawler"]
        for w in wordslist:

            if w not in stop_words and w not in custom:
                meaningful_words.append(w)

        return meaningful_words

    # Task4

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
        print("l", lemmetized_word[1])

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
                print(k)
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


    def get_features(self):
        connection = pymysql.connect(host='localhost',
                             user='root',
                             db='research')
        cursor = connection.cursor()
        sql = 'SELECT reviewId,title,comment FROM `feature_or_improvment_request_data'
        cursor.execute(sql)
        result = cursor.fetchall()
        f=pd.DataFrame(result,columns=['r_id','title','comment'])
        f['merged']=f['title'].astype(str) + f['comment']
        processed=self.data_cleaning(f['merged'])
        #un=self.unique_word_frequency(processed)
        #m_w=self.most_freq_words(un)
        #print(m_w)
        l=['pdf','view','pdf','watch','pdf','upload']
        colc=self.collocation(processed)

        return colc




r=get_Features()
print(r.get_features())