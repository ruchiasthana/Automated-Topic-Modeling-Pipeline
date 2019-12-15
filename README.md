# Automated Topic Modeling Pipeline For  Conversational Data 
## Advanced Python for Data Science (Final Project Fall 2019)

[![Build Status](https://travis-ci.com/csci-e-29/2019fa-final-project-ruchiasthana.svg?token=LoHckPFosYy1y1PJ2eXw&branch=master)](https://travis-ci.com/csci-e-29/2019fa-final-project-ruchiasthana)

[![Maintainability](https://api.codeclimate.com/v1/badges/32e9444157afa590b963/maintainability)](https://codeclimate.com/repos/5df505699ce2d901a3007ae3/maintainability)

[![Test Coverage](https://api.codeclimate.com/v1/badges/32e9444157afa590b963/test_coverage)](https://codeclimate.com/repos/5df505699ce2d901a3007ae3/test_coverage)

**Table of Contents**
Imsert a Table of Contents Here 
- [Project Description](#project-description)
- [Course Content Integration](#course-content-integration)
- [Data Preprocessing](#data-preprocessing)
- [Topic Modeling](#topic-modeling)
- [NGram Analysis](#ngram-analysis)
- [Word Cloud](#word-cloud)
- [Technical Pipeline](#technical-pipeline)
- [Results and Conclusions](#results-and-conclusions)
- [Improvements](#improvements)

## Project Description
I work as a machine learning engineer at IBM. My team handles inbound communication channels for product specific webpages through platforms like chat. Our primary clients are internal IBM product teams looking to increase the accessibility of their product for customers. The chatbots we launch on IBM product specific webpages accure a lot of conversation data that can render meaningful insights about the products performance on the market. Below are key iterms that have a lot of value to our stakeholders:
1. A Determinination of the Most Frequently Asked Topics
2. Identification of the Most Commonly Mentioned Terms and Services 
4. Effective Visualizations that can efficiently and effectively communicate key insights to Non-Technical Project Managers and Stakeholders
For my final project I created an automated pipeline that fetches data from a cloud instance, cleans the data, and performs Topic Modeling, NGram Analysis, and Word Cloud Generation. You can view this process with the following command

```
pipenv run python3 -m final_project -a
```

@Developer Note: Since Topic Modeling can take a very long time, I have included an updated TopicModeling output file with my submission. Please feel free to add this file to : './output/TopicModeling/' and then run above following command to render the automated pipeline more quickly. Below I will detail each process individually along with commands to run each part in isolation!

### Course Content Integration
Before diving into a detailed report of my final project, I just wanted to highlight that this project touches on a number of topics we have learned over the course of this class including: 

 - Setting up an EC2 instance and Loading Data into Amazon EC2 Instance 
 - Using Luigi to Pull Data from an Amazon EC2 Instance 
 - Using Dask to Clean Data in a Delayed, Parallel Fashion 
 - Working with and Extending an NLP Library (WordEmbeddings PSET 4)
 - Using Pandas to efficiently analyze a dataframe of healdlines
 - Class and Inheritance Patterns to Build Classes for Topic Modeling, NGram Analysis, and WordCloud Classes
 - Lambda Functions to apply Topic Modeling / NGram / WordCloud functions in a Parallel Manner 
 - Testing Pipeline and Workflow, as well as Task Functionality 
 - Salting the Topic Modeling Output Path so that the model is only run if the training data changed
 - Adding an argument parser, cli file, and main file
 - Creating a package with: data folder, output folder, and task folder 
 - Git Quality : Doc Strings, Pipfile, Piplockfile, travis.yaml 
 - Python Quality: Strong Commit History, Dev Branch, and Versioning


## Data Preprocessing
For the Client POC I build, I used historical chat transcript data, however that is not something I can share on this platform. For that purpose, I have loaded Arctile Text onto an AWS EC2 instance. Article Text was chosen because it is similar in nature to the chatbot utterances we wanted to analyze with Topic Modeling. Furthermore, the dataset was similar in size and dimension to the chat transcript data we have. 

Data was loaded onto the AWS EC2 instance, it waas then cleaned by a CleanedHeadlines class before generating a Topic Model. Pre-processing the data included removing columns with empty or null data, as well as removing duplicates in data. Article ID Tags came in super handy for this process. Once the data was cleaned it was stored in partioned *.parquet files in the ./data/CleanedHeadlines-[date] folder. The reason behind including the date in the folders was to ensure we were only fetching data once a day. This was a descision made from our stakholders. 

If you want to see the sole behavior of the Data Processing please run the command below: 
```
pipenv run python3 -m final_project -cr
```
This will generate .parquet files in the ./data/CleanedHeadlines-[date] folder.

## Topic Modeling
Topic modeling is a type of statistical modeling that can be used to discover abstract “topics” that occur in a collection of documents. This method could be applied to our chat trascripts to determine what topics are commonly mentioned in chat conversations. Latent Dirichlet Allocation (LDA) is an example of a topic model that is used to classify text in a document to a particular topic. It builds a topic per document model and words per topic model, modeled as Dirichlet distributions. We can take the topics generated by Topic Modeling and use them to classify each chat conversation. Tagging a conversation with a generated topic can be a good way to show the variety of utterances captured under a generated topic. For the sake of demonstration, this process was applied to Article Text.

When exploring Topic Modeling, we wanted to compare different models to see which would perform better. The two models in particular was an LDA and LDA + TFIF model. The pre-processing and post-processing for both topic modeling approaches was the same, and the structure for the models was also quite similar. Using Class and Inhertiance patterns I was able to create a Generalized Topic Modeling class that could be inherited by both methods. Below are a set of key functions that make up the this Topic Modeling Class. 

```
class TopicModeling(self):
    def load_data(self):
        ...

    def preprocess_document(self, text):
        ...

    def preproces_all_documents(self, data_file, field_name, output_file):
        ...
```

If you want to see the sole behavior of Topic Modeling please run the command below: 
```
pipenv run python3 -m final_project -tm
```
The output of this function will be stored in output/TopicModeling/HeadlineText-[hash].csv

### Salted Output for Topic Modeling
Running the Topic Modeling Code can be a very time cosuming process taking upwards of 20 minutes. While this time constraint is something our stakeholders seem to be okay with, we do not want them waiting around for a Topic Modeling Script to run if the cleaned data has not changed. To avoid this, we concatonate the last 50 headline_ids and hash the content to generate a len 8-alphanumeric id that we append to the output filepath. If an outputfile with the len 8-alphanumeric id exists, we do not re-run the topic modeling script as we know the data has not changed. 

## NGram Analysis 
We also wanted to provide our stakeholders with some specific insight on terms that customers strongly expressed when interacting with an IBM product. In order to do this, we conducted the NGram Analysis. N-Grams can be a very useful tool when trying to figure out which words and phrases are commonly expressed in a set of unstructured data. Analyzing trends in N-Grams can tell us what topics customers have started talking about, and what topics have fallen out of favor. We focused mainly on 2 and 3 letter phrases known as bigrams and trigrams for this analysis. TF-IDF weightings can be applied to sets of ngrams to narrow down the scope of bigrams and trigrams returned. 

Once we generated top bigrams and trigrams from our dataset, we went back and tagged each utterance with the most related bigrams and trigrams given a certain threshold. Our stakeholder's really liked this feature because it allowed them to take a commonly expressed bigram and see which utterances mentioned them. 

If you want to see the sole behavior of the NGram Analysis please run the command below: 
```
pipenv run python3 -m final_project -ng
```
The output of this function will be stored in output/UtteranceNGramMappings/HeadlineText-[hash].csv


## Word Cloud
When you are parsing through conversations ranging in the millions, it is often diffuclt to convey the resulsts in an effective visual to both technical and non-technical adopters. Glancing at a Word Cloud, one could easily capture the most frequently occuring keywords in a set of data. A word cloud had a strong info to ink ratio making it an exceptional visual in the NLP community. I was able to create a Word Cloud based off terms in the conversation / headline text. Our product stakeholders really liked this visualization as it was "effecient, effective, and highly non-technical". The template to run this method is shown below:

If you want to see the sole behavior of the WordCloud Generator please run the command below: 
```
pipenv run python3 -m final_project -wc
```
The output of this function will be stored in output/UtteranceNGramMappings/HeadlineText-[hash].png

![HeadlineText-0c66ad05](https://user-images.githubusercontent.com/42304193/70844344-e5fa0f80-1e0d-11ea-8e57-010d263f70c7.png)

## Results and Conclusions 
In conclusion I was able to automated a pipeline that could run TopicModeling, NGram Analysis, and Word Cloud Generation on a set of Conversational Text. Below is a representation of this pipeline: 

<img width="1257" alt="pipeline" src="https://user-images.githubusercontent.com/42304193/70844356-0fb33680-1e0e-11ea-837c-53e8f52dd2fb.png">

In order to run the complete pipeline you can run the following command:

```
pipenv run python3 -m final_project -a 
```

## Improvements
The stakeholders for this project were very impressed with the results from Topic Modeling. Together we were able to outline some next steps to proceed with the project in the next business quarter: 

1. Integrate Pipeline with Cloudant Database
2. Add Visualizations that Communicate Valuable Insights
3. Containerize the pipeline (Consider Kubflow)
4. Add more extensive unit tests

