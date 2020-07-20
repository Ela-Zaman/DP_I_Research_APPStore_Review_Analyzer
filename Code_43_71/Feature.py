import pandas as pd
import pymysql
from nltk.collocations import *
from nltk.metrics.association import *
from nltk.corpus import brown


from nltk.corpus import wordnet
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

        return r




    def collocation(self,r):

        # collocation
        feature = []
        #for i in r:

        #bcf = BigramCollocationFinder.from_words(brown.i()[:2000])
        bgm = nltk.collocations.BigramAssocMeasures()
            #print(i)
        for i in range(0,len(r)):
            finder = BigramCollocationFinder.from_words(r[i])
            #scored = finder.score_ngrams(bgm.likelihood_ratio)
            l = finder.nbest(BigramAssocMeasures.likelihood_ratio,40)
            feature.append(l)


        word_frequency = {}
        for words in feature:
            for w in words:
                if w not in word_frequency.keys():
                    word_frequency[w] = 1


                else:
                    word_frequency[w] += 1
        f_list = []
        r = []
        for k in word_frequency.keys():
            if word_frequency[k] > 1:
                print(k)
                f_list.append([k, word_frequency[k]])

    # word_synonyms = get_word_synonyms_from_sent(word, sent)
    # print ("WORD:", word)
    # print ("SENTENCE:", sent)


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
        print(processed)
        colc=self.collocation(processed)




r=get_Features()
r.get_features()