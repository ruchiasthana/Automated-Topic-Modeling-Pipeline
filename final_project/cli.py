from luigi import build
from final_project.tasks.clean_headlines import ArticleHeadlines, CleanedHeadlines
from final_project.tasks.topic_modeling import ApplyTopicModel
from final_project.tasks.word_cloud import CreateWordCloud
from final_project.tasks.ngram_analysis import EnableFuzzyMatching

# from pset_5.tasks.yelp_reviews import ByDecade, ByStars
import argparse
import time 
import datetime 

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--full", action="store_false", dest="full")
parser.add_argument("-cr", "--cleanedreviews", action="store_true")
parser.add_argument("-tm", "--tpoicmodeling" , action="store_true")
parser.add_argument("-ng", "--ngram" , action="store_true")
parser.add_argument("-wc", "--wordcloud", action="store_true")
parser.add_argument("-a", "--all", action="store_true")



def main(args=None):
    args = parser.parse_args(args=args)
    print('Arguments', args)
    #When testing you may only want to run a subset of the fully pipeline, the argument parser 
    #Allows you to do just that!
    
    run_cr = args.cleanedreviews 
    run_tm = args.tpoicmodeling
    run_ng = args.ngram
    run_wc = args.wordcloud
    run_all = args.all


    #This is the case where you only want to run CleanedReviews to See the Data 
    if (run_cr == True) and (run_tm == False) and (run_ng == False) and (run_wc == False):
        start = time.time()
        date = datetime.datetime.now()
        date_suffix = str(date.month) + '_' + str(date.day) + '_' + str(date.year)
        build([CleanedHeadlines(subset=False)],local_scheduler=True)
        end = time.time()
        print('The total time for running Cleaned Reviews: ', end-start, 's')
    
    #This is the case where you only want to run Topic Modeling 
    elif (run_tm == True) and (run_ng == False) and (run_wc == False):
        start = time.time()
        date = datetime.datetime.now()
        date_suffix = str(date.month) + '_' + str(date.day) + '_' + str(date.year)
        print("\n", "Welcome User! You will be running Topic Modeling Analysis today! The date of this analysis is: ", date_suffix, "\n")
        print("Please note that Topic Modeling is a Time Consuming Process. It can take upwards of 15min to create a Topic Model.")
        
        #You need to fetch the most current data before you can do Topic Modeling!
        build([CleanedHeadlines(subset=False)],local_scheduler=True)
        data_outputfolder = "./data/CleanedHeadlines-" + date_suffix  + "/"
        
        print("\n", "Creating a Topic Model that will be stored at: ", data_outputfolder, "\n")
        model = ApplyTopicModel(
            data_folder = data_outputfolder,
            data_filetype = 'parquet',
            model_type = 'lda_tfidf',
            field_name = 'headline_text'
        )
        model.run()
        model_output = model.output_file
        end = time.time()
        print('The model output has been created! You can see it here:', model_output)
        print('The total time for running Topic Modeling: ', end-start, 's')
    
    #NGram Anlysis Needs an Updated Topic Model 
    elif (run_ng == True) and (run_wc == False):
        start = time.time()
        date = datetime.datetime.now()
        date_suffix = str(date.month) + '_' + str(date.day) + '_' + str(date.year)
        print("\n", "Welcome User! You will be running Topic Modeling Analysis today! The date of this analysis is: ", date_suffix, "\n")
        build([CleanedHeadlines(subset=False)],local_scheduler=True)
        data_outputfolder = "./data/CleanedHeadlines-" + date_suffix  + "/"
        
        print("\n", "Creating a Topic Model that will be stored at: ", data_outputfolder, "\n")
        model = ApplyTopicModel(
            data_folder = data_outputfolder,
            data_filetype = 'parquet',
            model_type = 'lda_tfidf',
            field_name = 'headline_text'
        )
        model.run()
        model_output = model.output_file
        print('The model output has been created! You can see it here:', model_output)
        print("\n", "So far we have analyzed key topics and generated representative topic labels for our utterances.", 
            "\n", "Now we will dive into Keyword Analysis. Our anlysis will return top bigrams and trigrams from the text along with \n extracted topics for each utterance")

        fm = EnableFuzzyMatching(
            data_folder = data_outputfolder,
            data_filetype = 'parquet',
            field_name = 'headline_text',
            utterance_ngram_mapping_output= model_output.replace('TopicModeling', 'UtteranceNGramMappings')
        )
        fm.write_ngrams_to_utterances()
        end = time.time()
        print('The total time for running NGram Analysis: ', end-start, 's')
    
    elif (run_ng == False) and (run_wc == True):
        start = time.time()
        date = datetime.datetime.now()
        date_suffix = str(date.month) + '_' + str(date.day) + '_' + str(date.year)
        print("\n", "Welcome User! You will be running Topic Modeling Analysis today! The date of this analysis is: ", date_suffix, "\n")
        
        build([CleanedHeadlines(subset=False)],local_scheduler=True)
        data_outputfolder = "./data/CleanedHeadlines-" + date_suffix  + "/"
        
        print("\n", "Creating a Topic Model that will be stored at: ", data_outputfolder, "\n")
        model = ApplyTopicModel(
            data_folder = data_outputfolder,
            data_filetype = 'parquet',
            model_type = 'lda_tfidf',
            field_name = 'headline_text'
        )
        model.run()
        model_output = model.output_file
        print('The model output has been created! You can see it here:', model_output)
        print('Visualizing Data is Always Helpful. We can do that by Generating a Word Cloud for Important Keywords and Phrases!')
        print('~~~~~~~Generating Word Cloud~~~~~~~')
        cloud = CreateWordCloud(
            text_path = model_output,
            field_name = 'headline_text',
            output_path = model_output.replace('TopicModeling', 'WordClouds').replace('.csv', '.png')
        )
        cloud.create_wordcloud()
        print('Word Cloud has been Generated! You can find it here: ', model_output.replace('TopicModeling', 'WordClouds'))
        end = time.time()
        print('The total time for running Word Cloud Generation: ', end-start, 's')

    elif ((run_ng == True) and (run_wc == True)) or (run_all == True):
        start = time.time()
        date = datetime.datetime.now()
        date_suffix = str(date.month) + '_' + str(date.day) + '_' + str(date.year)
        print("\n", "Welcome User! You will be running Topic Modeling Analysis today! The date of this analysis is: ", date_suffix, "\n")
        build([CleanedHeadlines(subset=False)],local_scheduler=True)
        data_outputfolder = "./data/CleanedHeadlines-" + date_suffix  + "/"
        
        print("\n", "Creating a Topic Model that will be stored at: ", data_outputfolder, "\n")
        model = ApplyTopicModel(
            data_folder = data_outputfolder,
            data_filetype = 'parquet',
            model_type = 'lda_tfidf',
            field_name = 'headline_text'
        )
        model.run()
        model_output = model.output_file
        print('The model output has been created! You can see it here:', model_output)
        print("\n", "So far we have analyzed key topics and generated representative topic labels for our utterances.", 
            "\n", "Now we will dive into Keyword Analysis. Our anlysis will return top bigrams and trigrams from the text along with \n extracted topics for each utterance")

        fm = EnableFuzzyMatching(
            data_folder = data_outputfolder,
            data_filetype = 'parquet',
            field_name = 'headline_text',
            utterance_ngram_mapping_output= model_output.replace('TopicModeling', 'UtteranceNGramMappings')
        )
        fm.write_ngrams_to_utterances()

        print('Visualizing Data is Always Helpful. We can do that by Generating a Word Cloud for Important Keywords and Phrases!')
        print('~~~~~~~Generating Word Cloud~~~~~~~')
        cloud = CreateWordCloud(
            text_path = model_output,
            field_name = 'headline_text',
            output_path = model_output.replace('TopicModeling', 'WordClouds').replace('.csv', '.png')
        )
        cloud.create_wordcloud()
        print('Word Cloud has been Generated! You can find it here: ', model_output.replace('TopicModeling', 'WordClouds'))
        
        end = time.time()
        print('Total Time for Running Cleaned Reviews, Topic Modeling, NGram Analysis, and WordCloud', end-start,'s')


    
