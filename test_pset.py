import os
import datetime
import luigi
import numpy as np
import pandas as pd
from unittest import TestCase
from luigi import LuigiStatusCode
from tempfile import TemporaryDirectory
from luigi import ExternalTask, Parameter, Task
from final_project.tasks.clean_headlines import CleanedHeadlines


"""
These first set of test are to ensure that we can 
1. Create a bucket in AWS
2. Access data from the bucket creaated in AWS
3. Download content for the created bucket, and store them in TmpDir Locally
Template for the first Test Case comes from: 
    https://medium.com/@l.peppoloni/how-to-mock-s3-services-in-python-tests-dd5851842946
"""

class TestPipeline(TestCase):
    # def testCleanedHeadlinesBuild(self):
    #     build_status = luigi.build([CleanedHeadlines(subset=False)],local_scheduler=True, detailed_summary=True).status
    #     self.assertEqual(build_status, LuigiStatusCode.SUCCESS)

    # def testCleanedHeadlinesFileExists(self):
    #     build_status = luigi.build(
    #         [CleanedHeadlines()], local_scheduler=True, detailed_summary=True
    #     ).status
    #     date = datetime.datetime.now()
    #     date_suffix = str(date.month) + '_' + str(date.day) + '_' + str(date.year)
    #     print('OUTPUT PATH', "./data/CleanedHeadlines-" + date_suffix + "/")
    #     self.assertTrue(os.path.exists("./data/CleanedHeadlines-" + date_suffix + "/"))
    #     self.assertTrue(os.path.exists("./data/CleanedHeadlines-" + date_suffix + "/part.0.parquet"))

    def testDropNA(self):
        data = [['Tom', 10], ['Jen',5], ['Ruchi', np.nan]]
        df = pd.DataFrame(data, columns=['Name', 'Age'])
        df = df.dropna()
        self.assertTrue('Tom' in df['Name'].tolist())
        self.assertTrue('Jen' in df['Name'].tolist())
        self.assertFalse('Ruchi' in df['Name'].tolist())
    
    def testGroupBy(self):
        data = [['Tom', 10, 2010], ['Jen',5, 2010], ['Ruchi', 7, 2011]]
        df = pd.DataFrame(data, columns=['Name', 'Age', 'Year'])
        df_groupBy = df.groupby('Year').mean()
        self.assertTrue(df_groupBy['Age'][2010], 7.5)
        self.assertTrue(df_groupBy['Age'][2011], 7.0)