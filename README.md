# Automated Topic Modeling Pipeline For  Conversational Data 
## Advanced Python for Data Science (Final Project Fall 2019)

[![Build Status](https://travis-ci.com/csci-e-29/2019fa-pset-5-ruchiasthana.svg?token=LoHckPFosYy1y1PJ2eXw&branch=master)](https://travis-ci.com/csci-e-29/2019fa-pset-5-ruchiasthana)

[![Maintainability](https://api.codeclimate.com/v1/badges/32e9444157afa590b963/maintainability)](https://codeclimate.com/repos/5df505699ce2d901a3007ae3/maintainability)

[![Test Coverage](https://api.codeclimate.com/v1/badges/32e9444157afa590b963/test_coverage)](https://codeclimate.com/repos/5df505699ce2d901a3007ae3/test_coverage)

**Table of Contents**
Imsert a Table of Contents Here 
- [Project Description](#project-description)
- [Data Preprocessing](#data-preprocessing)
- [Topic Modeling](#topic-modeling)
- [NGram Analysis](#ngram-analysis)
- [Technical Pipeline](#technical-pipeline)
- [Results and Conclusions](#results-and-conclusions)
- [Improvements](#improvements)

## Project Description
I work as a machine learning engineer at IBM. My team handles inbound communication channels for product specific webpages through platforms like chat. Our primary clients are internal IBM product teams looking to increase the accessibility of their product for customers. The chatbots we launch on IBM product specific webpages accure a lot of conversation data that can render meaningful insights about the products performance on the market. Below are key iterms that have a lot of value to our stakeholders:
1. A Determinination of the Most Frequently Asked Topics
2. Identification of the Most Commonly Mentioned Terms and Services 
4. Effective Visualizations that can efficiently and effectively communicate key insights to Non-Technical Project Managers and Stakeholders

### Course Content Integration
For my final project I created an automated pipeline that fetches data from a cloud instance, cleans the data, and performs Topic Modeling, NGram Analysis, and Word Cloud Generation. The project touches on a number of topics we have learned over the course of this class including: 
 - Setting up an EC2 instance and Loading Data into Amazon EC2 Instance 
 - Using Luigi to Pull Data from the Amazon EC2 Instance 
 - Using Dask to Clean Data in a Delayed, Parallel Fashion 
 - Working with and Extending an NLP Library (WordEmbeddings PSET 4)
 - Using Pandas to efficiently analyze dataframe of healdlines
 - Classes and Inheritance Patterns to Build the Topic Modeling, NGram Analysis, and WordCloud Classes
 - Lambda Functions to Appply Topic Modeling / NGram / WordCloud functions in a Parallel Manner 
 - Testing Pipeline and Workflow, as well as Task Functionality 
 - Package Structure and Heirachy with data, output, final_project, and tasks folders 
 - Git Quality : Doc Strings, Pipfile, Piplockfile, travis.yaml 
 - Python Quality: Strong Commit History, Dev Branch, Versioning


## Data Preprocessing
For the Client POC I build, I used historical chat transcript data, however that is not something I can share on this platform. For that purpose, I have loaded Arctile Text onto an AWS EC2 instance. Article Text was chosen because it is similar in nature to the chatbot utterances we wanted to analyze with Topic Modeling. 
Data Source : 

Data tha was loaded into the AWS EC2 instance needed to be cleaned before Topic Modeling could be run on the data set. Pre-processing the data included removing columns with empty or null data, as well as removing duplicates in data. Article ID Tags came in super handy for this process. Once the data was cleaned it was stored in partioned *.parquet files in the ./data/CleanedHeadlines-[date] folder. The reason behind including the date in the folders we create to store the data is so that we are only fetching data once a day. This was a descision made from our stakholders. 

If you want to run code to only generate the ./data/CleanedHeadlines-[date] folder run the following command: 

```
pipenv run python3 final_project/tasks/cleaned_headlines.py
```

## Topic Modeling
Topic modeling is a type of statistical modeling that can be used to discover abstract “topics” that occur in a collection of documents. This methods could be applied to our chat trascripts to determine what topics are commonly mentioned in chat conversations. Latent Dirichlet Allocation (LDA) is an example of a topic model that is used to classify text in a document to a particular topic. It builds a topic per document model and words per topic model, modeled as Dirichlet distributions. We can take the topics generated by Topic Modeling and use them to classify each chat conversation. Tagging a conversation with a generated topic can be a good way to show the variety of utterances captured under a generated topic.

When conducting our analysis for Topic Modeling we wanted to consider 2 models in particular: LDA and LDA+TFIDF. The pre-processing and post-processing, as well as model structure is the same for both of these models, so I decided to create a Topic Modeling Class that could be shared between our investigation of both methods. Below are a subset of key functions that make up the Topic Modeling class.
```
class TopicModeling(self):
    def load_data(self):
        ...

    def preprocess_document(self, text):
        ...

    def preproces_all_documents(self, data_file, field_name, output_file):
        ...
```

The structure for calling the Topic Modeling Class is show below. Note that the data_outputfolder should point to the path where the CleanedHeadlines are. 

```
model = ApplyTopicModel(
        data_folder = cleaned_data_outputfolder,
        data_filetype = 'parquet',
        model_type = 'lda_tfidf',
        field_name = 'headline_text'
    )
``` 

### Logistics of Topic Modeling (SALTING!)
Once we have cleaned the data and loaded it into the ./data/CleanedHeadlines-[date] folder we want to conduct Topic Modeling in the whole data set if new data is added. Since Topic Modeling is a very time cosuming task, we wanted to conduct Topic Modeling only when there was new data. Thus we collected that last 50 headline_id values concatenedated them into a string and hashed that string with a salt to get a 8-alphanumeric id that we could append to the Topic Modeling output folder. If a Folder with the 8-alphanumeric id exists in the ./output/TopicModeling folder we return the path to that folder, otherwise we run the Topic Modeling Script. 

## NGram Analysis 
In addition to Topic Modeling, we also wanted to provide additional more specific insight into ket terms strongly expressed in our data set. N-Grams can be a very useful tool when trying to figure out which words and phrases are commonly expressed in a set of unstructured data. Analyzing trends in N-Grams can tell us what topics customers have started talking about, and what topics that have fallen out of favor. TF-IDF weightings can be applied to N-Gram extraction to narrow down the scope of bigrams and trigrams. 

Once we generated top bigrams and trigrams from our dataset, we went back and tagged each utterance with bigrams and trigrams if they were in the sentence or vernacular in the sentenec closely matched the ngrams extracted. The template to run this method is shown below:

```
fm = EnableFuzzyMatching(
        data_folder = data_outputfolder,
        data_filetype = 'parquet',
        field_name = 'headline_text',
        utterance_ngram_mapping_output= model_output.replace('TopicModeling', 'UtteranceNGramMappings')
    )
    fm.write_ngrams_to_utterances()

```

After writing a class to(1) determine top Bigrams and Trigrams in our dataset, and (2) labele Article Headlines with closely matched bigrams and trigrams, I was able to create a Word Cloud of the terms that could in a glance communicate these results. Our product stakeholders really liked this visualization as it was "effecient, effective, and highly non-technical". The template to run this method is shown below:

```
cloud = CreateWordCloud(
        text_path = model_output,
        field_name = 'headline_text',
        output_path = model_output.replace('TopicModeling', 'WordClouds').replace('.csv', '.png')
    )
    cloud.create_wordcloud()
```

![Sample Word Cloud]("images/HeadlineText-0c66ad05.png")

## Results and Conclusions 
In conclusion I was able to automated a pipeline that could run TopicModeling, NGram Analysis, and Word Cloud Generation on a set of Conversational Text. Below is a representation of this pipeline: 

![Automated Topic Modeling Pipeline]("images/pipeline.png")

Below is the command to run this automated pipepline : 

```
pipenv run python3 -m final_project
```

## Improvements
The stakeholders for this project were very impressed with the results from Topic Modeling. Together we were able to outline some next steps to proceed with the project in the next business quarter: 

1. Integrate Pipeline with Cloudant Database
2. Add Visualizations that Communicate Valuable Insights
3. Containerize the pipeline (Consider Kubflow)

