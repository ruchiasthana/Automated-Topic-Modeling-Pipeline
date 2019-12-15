import nltk 
import re
import glob
import functools
import itertools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import pandas as pd 
from fuzzywuzzy import fuzz 
from fuzzywuzzy import process 


class ApplyNGramAnalysis(object):

    '''
    TF-IDF Term Frequency-Inverse Document Frequency 
    Determines import terms in a document, wether it is bigrams or trigrams 
    Term Frequency (TF) = (Frequency of a term in the document)/(Total number of terms in documents)
    Inverse Document Frequency(IDF) = log( (total number of documents)/(number of documents with term t))
    TF.IDF = (TF).(IDF)
    '''

    def __init__(self, data_folder, data_filetype, field_name):
        self.data_folder = data_folder
        self.data_filetype = data_filetype
        self.field_name = field_name
        if self.data_filetype == 'csv':
            self.data_df = pd.concat(map(lambda x: pd.read_csv(x), glob.glob(data_folder+"*.csv")))
            self.data_df = self.data_df.head(50000)
        if self.data_filetype == 'parquet':
            self.data_df = pd.concat(map(lambda x: pd.read_parquet(x), glob.glob(data_folder+"*.parquet")))
            self.data_df = self.data_df.head(50000)
        self.bigrams = None 
        self.trigrams = None

    def aggregate_text(self):
        # data_df = pd.read_csv(self.data_file)
        all_data_text = self.data_df[self.field_name].tolist()
        return all_data_text
    
    def remove_special_chars(self, all_data_text):
        stripped = re.sub('[^a-zA-z\s]', '', all_data_text)
        stripped = re.sub('_', '', stripped)
        stripped = re.sub('\s+', ' ', stripped)
        stripped = stripped.strip() 
        if stripped != '':
            return stripped.lower()
    
    def get_topic_ngrams(self, ngram):
        all_data_text = self.aggregate_text()
        stop_words = set(stopwords.words('english'))
        all_data_text =  list(map(lambda line: ' '.join([x for x in nltk.word_tokenize(str(line)) if  (x not in stop_words)]), all_data_text))
        vectorizer = CountVectorizer(ngram_range = (ngram,ngram))
        X1 = vectorizer.fit_transform(all_data_text)
        features = (vectorizer.get_feature_names())
        
        # Applying TFIDF 
        vectorizer = TfidfVectorizer(ngram_range = (ngram,ngram)) 
        X2 = vectorizer.fit_transform(all_data_text) 
        
        # Getting top ranking features 
        sums = X2.sum(axis = 0) 
        data1 = [] 
        for col, term in enumerate(features): 
            data1.append( (term, sums[0,col] )) 
        ranking = pd.DataFrame(data1, columns = ['term','rank']) 
        words = (ranking.sort_values('rank', ascending = False))
        words = words[words['rank'] > 1]
        print ("\n\nWords head : \n", words.head(10)) 
        return words
    
    def load_bigrams(self):
        self.bigrams = self.get_topic_ngrams(ngram=2)['term'].tolist()
        return self.bigrams
    
    def load_trigrams(self):
        self.trigrams = self.get_topic_ngrams(ngram=3)['term'].tolist()
        return self.trigrams

class EnableFuzzyMatching(ApplyNGramAnalysis):
    def __init__(self,data_folder, data_filetype, field_name, utterance_ngram_mapping_output):
        ApplyNGramAnalysis.__init__(self, data_folder, data_filetype, field_name)
        if self.data_filetype == 'csv':
            self.data_df = pd.concat(map(lambda x: pd.read_csv(x), glob.glob(data_folder+"*.csv")))
            self.data_df = self.data_df.head(50000)
        if self.data_filetype == 'parquet':
            self.data_df = pd.concat(map(lambda x: pd.read_parquet(x), glob.glob(data_folder+"*.parquet"))) 
            self.data_df = self.data_df.head(50000)
        self.topic_to_ngram = None 
        self.ngram_to_topic = None 
        self.fuzzymatch_df = None
        print("\n", "~~~~~~~Loading Bigrams~~~~~~~")
        self.bigrams = ApplyNGramAnalysis.load_bigrams(self)
        print("\n", "~~~~~~~Loading Trigrams~~~~~~~")
        self.trigrams = ApplyNGramAnalysis.load_trigrams(self)
        self.utterance_ngram_mapping_output = utterance_ngram_mapping_output
        
    def invert_dict(self, og_dict):
        inverted_dict = dict()
        for key in og_dict.keys():
            values = og_dict[key]
            for value in values:
                inverted_dict[value] = key
        return inverted_dict
    
    def ngram_grouping(self, ngrams):
        sorted_ngrams = sorted(ngrams)
        #Hack for testing only 
        # sorted_ngrams = sorted_ngrams[:10]
        topic_to_ngram = dict()
        flatten = lambda l: [item for sublist in l for item in sublist]
        for idx in range(0, len(sorted_ngrams), 1):
            ngram = sorted_ngrams[idx]
            if not ngram in list(flatten(topic_to_ngram.values())):
                related_terms = list(filter(lambda x: fuzz.ratio(x, ngram) > 60, sorted_ngrams[idx:]))
                topic_to_ngram[ngram] = related_terms
        return topic_to_ngram
    
    def apply_ngrams_to_utterance(self, utterance):
        if self.bigrams == None or self.trigrams == None:
            self.bigrams = self.ngram_grouping(ngram=2)
            self.bigrams_flattened = list(itertools.chain(*self.bigrams.values()))
            self.trigrams = self.ngram_grouping(ngram=3)
            self.trigrams_flattened = list(itertools.chain(*self.trigrams.values()))
            self.topic_to_ngram = {**self.bigrams, **self.trigrams} 
            self.ngram_to_topic = self.invert_dict(self.topic_to_ngram)
        
        if len(self.bigrams) == 0 or len(self.trigrams) == 0:
            print ('You did not load the Ngrams - Load these values by calling the loadNGrams Function!')
        else:
            if type(utterance) is str: 
                bigram_matches = list(filter(lambda x: float(fuzz.ratio(utterance, x)) > 60, self.bigrams))
                trigram_matches = list(filter(lambda x: float(fuzz.ratio(utterance, x)) > 70, self.trigrams))
                return bigram_matches + trigram_matches
            else:
                return None 
    
    def write_ngrams_to_utterances(self):
        df = self.data_df
        df = df.head(50)
        print('\n', 'Applying NGram Mappings to Utterances')
        df['ngrams'] = df[self.field_name].apply(lambda x: self.apply_ngrams_to_utterance(x))
        print('You can find NGram Mappings Here:', self.utterance_ngram_mapping_output)
        df.to_csv(self.utterance_ngram_mapping_output)
        print(df.head(10))