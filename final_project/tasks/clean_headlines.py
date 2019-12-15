import os
import luigi
import glob
import datetime
import pandas as pd
from luigi.contrib.s3 import S3Target
from luigi.parameter import BoolParameter
from luigi import ExternalTask, Parameter, Task
from csci_utils.luigi.dask.target import CSVTarget
from csci_utils.hash_str.hash_str import get_user_id, hash_str, get_csci_salt
from csci_utils.luigi.task import Requirement, Requires, TargetOutput
from csci_utils.luigi.dask.target import ParquetTarget, CSVTarget, BaseDaskTarget



class ArticleHeadlines(ExternalTask):
    '''
    Article Headlines are Located on an Amazon EC2 Instance. This class will check to see that 
    a path to the Article Headlines data exists 

    :input : path to s3 article headlines
    :output : CSV Target if it exists
    '''

    def output(self):
        try:
            ##Check that this is a reasonable way to load the S3 Path 
            s3_path = os.environ["S3_PATH"]
            csv_target = CSVTarget(s3_path, flag=None, glob="*.csv")
            return csv_target
        except:
            ##Double check that this a reasonable way to Handle Exceptons 
            raise Exception("~~ The path to Headline Articles Data Does not Exist ~~")

class CleanedHeadlines(Task):
    '''
    This class loads the data from the AWS instance if it exists and preprocesses the data for analysis 
    The class returns a dataframe with pre-processed reviews that can be loaded in the Topic Modeling class 
    for additional analysis 

    :input : s3 Path to Article Headlines 
    :output : Creates a Local Parquet File with the preprocessed data 
    '''
    subset = BoolParameter(default=True)
    requires = Requires()
    article_headlines = Requirement(ArticleHeadlines)
    date = datetime.datetime.now()
    date_suffix = str(date.month) + '_' + str(date.day) + '_' + str(date.year)

    output = TargetOutput(target_class=ParquetTarget, ext='-'+date_suffix, glob="*.parquet",)
    def run(self):
        dsk = self.input()['article_headlines'].read_dask(
            dtype={
                "publish_date" : "int32",
                "headline_text" : "str",
                "headline_id" : "str"
            },
            storage_options=dict(requester_pays=True),)
        
        # dsk_df = dsk.compute()
        headlines_concat = "".join(dsk["headline_id"])
        headlines_hash = hash_str(headlines_concat, get_csci_salt()).hex()[:8] 
        self.output().write_dask(dsk, compression="gzip")
        
    def print_results(self):
        print(self.output().read_dask().compute())

if __name__ == "__main__":
    luigi.build([CleanedHeadlines(subset=False)],local_scheduler=True)
