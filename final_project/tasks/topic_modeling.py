import os 
import ast
import nltk
import glob
import gensim
import functools
import itertools
import numpy as np
import pandas as pd
from functools import reduce
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from csci_utils.hash_str.hash_str import get_user_hash, get_csci_salt

np.random.seed(2018)
nltk.download("wordnet")

class ApplyTopicModel(object):

    def __init__(self, data_folder, data_filetype, model_type, field_name):
        self.data_folder = data_folder
        self.data_filetype = data_filetype
        self.field_name = field_name
        if self.data_filetype == 'csv':
            self.unprocessed_df = pd.concat(map(lambda x: pd.read_csv(x), glob.glob(data_folder+"*.csv")))
        if self.data_filetype == 'parquet':
            self.unprocessed_df = pd.concat(map(lambda x: pd.read_parquet(x), glob.glob(data_folder+"*.parquet")))
        self.postprocessed_df = None 
        self.stemmed_dict=dict()
        self.lda_model = None
        self.lda_tfidf_model = None 
        self.dictionary = None
        self.model_type = model_type
        self.dictionary = None
        self.bow_corpus = None
        self.lda_score_threshold = 0.33
        self.lda_n_topics = 20
        self.postprocessed_fieldname = 'postprocessed_'+field_name
        self.output_file = None
        self.topicmodeling_df = None
         
    def preprocess_document(self, text):
        """
        We will preprocess the document with the follwing methods in order to go from 
        a raw utterance to one that is tokenized, lemmatized, and stemmed. Methods were 
        conducted in the following order:

        1. Toekization 
        2. Stop word removal 
        3. Lemmatization
        4. Stemming

        :input: raw utterance
        :dtype: str

        :output: stemmed utterance
        :dtype: str
        """
        ss = SnowballStemmer("english")
        tokenized = word_tokenize(text)
        tokenized = list(filter(lambda x: x not in STOPWORDS, tokenized))
        tokenized = list(set(tokenized))
        tokenized = list(map(lambda x: x.lower(), tokenized))
        lemmatized = list(
            map(lambda x: WordNetLemmatizer().lemmatize(x, pos="v"), tokenized)
        )
        stemmed = list(map(lambda x: ss.stem(x), lemmatized))
        zipped_list = list(zip(stemmed, tokenized))
        for zip_elem in zipped_list:
            (stem, token) = zip_elem
            if (stem != token) and (stem not in self.stemmed_dict.keys()):
                self.stemmed_dict[stem] = [str(token)]
        return stemmed
    
    def preprocess_all_documents(self):
        """
        preprocess_document pre-processes each utterance that is used to build the Topic Model.
        preprocess_all_documents applies the preprocess_document function to all utterances that are in the training data for the Topic Model.
        We use lambda functions to parallelize this code, and increase the efficiency by which all data is pre-processed

        :input: data_file, field_name, output_file
        :dtype: str (for a .csv file), str, str (for a .csv file)

        :output: processed_df 
        :dtype: pandas DataFrame, (.csv)
        """
        df = self.unprocessed_df
        (rows, cols) = df.shape
        processed_df = df[self.field_name].apply(lambda x: self.preprocess_document(str(x)))
        df[self.postprocessed_fieldname] = processed_df
        self.postprocessed_df = df 
        return df 
    
    def lda_model(self, bow_corpus, dictionary):
        lda_model = gensim.models.LdaMulticore(
            bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2
        )
        self.lda_model = lda_model
        self.dictionary = dictionary
        return lda_model

    def lda_tfidf_model(self, bow_corpus, dictionary):
        model_params = dict()
        tfidf = models.TfidfModel(self.bow_corpus)
        corpus_tfidf = tfidf[self.bow_corpus]
        lda_model_tfidf = gensim.models.LdaMulticore(
            corpus_tfidf, num_topics=10, id2word=self.dictionary, passes=2, workers=4
        )
        self.lda_tfidf_model = lda_model_tfidf
        return lda_model_tfidf

    def generate_topic_model(self):
        """
        This function takes in utterances from bot conversations (dervied from the processed_df and field_name)
        it then uses these utterances to create a Topic Model that can be applied to unseen utterances to get a 
        topic vector of words and weights for the unseen utterance. We can use this topic vector to get keywords
        assoicated with the unseen utterance.

        :input: processed_df, field_name
        :dtype: pandas DataFrame, str

        :output: topic model 
        """
        #Creates a dictionary from processed_df that contains the number of times a word appears in the training set
        # self.postprocessed_df = self.postprocessed_df.head(50)
        postprocessed_df = list(self.postprocessed_df[self.postprocessed_fieldname])
        dictionary = gensim.corpora.Dictionary(postprocessed_df)
        dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        
        self.dictionary = dictionary
        
        """
        self.dictionary = gensim.corpora.Dictionary(self.postprocessed_df[self.postprocessed_fieldname])
        self.dictionary = self.dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        The bow_corpus is a list of list, where each element in the list is a document(utterance) 
        Each values in the bow_corpus represents the number of time a particular term occurs ]
        self.bow_corpus = self.reveal_bow_corpus(self.postprocessed_df, self.postprocessed_fieldname, self.dictionary)
        """

        self.bow_corpus = [self.dictionary.doc2bow(doc) for doc in self.postprocessed_df[self.postprocessed_fieldname]]
        
        if self.model_type == "lda_tfidf":

            tfidf = models.TfidfModel(self.bow_corpus)
            corpus_tfidf = tfidf[self.bow_corpus]
            lda_model_tfidf = gensim.models.LdaMulticore(
                corpus_tfidf, num_topics=10, id2word=self.dictionary, passes=2, workers=4
            )
            self.lda_tfidf_model = lda_model_tfidf

        else:
            lda_model = gensim.models.LdaMulticore(
            self.bow_corpus, num_topics=10, id2word=self.dictionary, passes=2, workers=2
            )
            self.lda_model = lda_model
            
    
    def apply_model(self, utterance):
        """
        This function takes a Topic Model and a new (unseen) utterance and it returns
        1. top Topics - vector of words and weights that represent the corpus of bot conversations
        2. a Model - takes an unseen utterance and returns n Topic vectors of words and weights

        :input: unseen_doc, model, n, dictionary
        :output: scores and topics
        """
        scores_topics = []
        bow_vector = self.dictionary.doc2bow(self.preprocess_document(utterance))
        if self.model_type == 'lda_tfidf':
            model = self.lda_tfidf_model
        elif self.model_type == 'lda':
            model = self.lda_model
        for index, score in sorted(model[bow_vector], key=lambda tup: -1 * tup[1]): 
            if score > self.lda_score_threshold:
                scores_topics.append(model.print_topic(index, self.lda_n_topics))
        return scores_topics
    
    def replaceStemWithWord(self, stem):
        if stem in self.stemmed_dict.keys():
            return self.stemmed_dict[stem][0]
        else:
            return stem
    
    def getTopicLabel(self, topic_list):
        topic_label = reduce((lambda x, y: x + "_" + y), topic_list)
        return topic_label
    
    def analyze_model(self, lda_applied):
        topic_label_list = []
        stemmed_key_words_list = []
        topic_dict = dict()
        if len(lda_applied) == 0: 
            return []
        else:
            for idx in range(0, len(lda_applied), 1):
                first_topic_vector = lda_applied[idx]
                first_topic_vector = first_topic_vector.replace('"','').replace(' ', '')
                split_by_plus = first_topic_vector.split('+')
                split_by_star = list(map(lambda x: x.split('*'), split_by_plus))
                score_to_float = list(map(lambda x: (float(x[0]), x[1]), split_by_star))
                sorted_scores = sorted(score_to_float)[::-1]
                sorted_key_words = list(map(lambda x: x[1], sorted_scores))
                filtered_key_words = list(filter(lambda x: len(x) > 3, sorted_key_words))
                stemmed_key_words = list(map(lambda x: self.replaceStemWithWord(x), filtered_key_words))
                stemmed_key_words_list.append(stemmed_key_words)
                topic_label = self.getTopicLabel(stemmed_key_words[:5])
                topic_label_list.append(topic_label)
            return topic_label_list

    def apply_model_to_utterance(self, utterance):
        lda_applied = self.apply_model(utterance)
        lda_analyzed = self.analyze_model(lda_applied)
        return lda_analyzed
    
    def run(self):
        df = self.unprocessed_df
        salt = get_csci_salt()
        last50_headlines = functools.reduce(lambda x,y: x+y, df[self.field_name].tail(100).tolist())
        hashed_id = get_user_hash(last50_headlines, salt=None).hex()[:8]
        output_file = './output/TopicModeling/HeadlineText-'+hashed_id+'.csv'
        self.output_file = output_file
        
        # If an output file with the hashed_id exists that means no data changed, so we do not 
        # need to run the time consuming preprocessing of documents and topic modeling script 
        if not os.path.exists(output_file):
            print('Pre-processing all documents')
            self.preprocess_all_documents()
            print('Generating a Topic Model')
            self.generate_topic_model()
            self.postprocessed_df['topic_modeling'] = self.postprocessed_df[self.field_name].apply(lambda x: self.apply_model_to_utterance(str(x)))
            print('Writing the Topic Model Output to: ', output_file)
            self.postprocessed_df.to_csv(output_file)
        else:
            print('The topic model has already been applied to your data. You can find it located here:', output_file)
    
    def output_file(self):
        return self.output_file