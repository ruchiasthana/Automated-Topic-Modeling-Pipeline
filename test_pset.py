import os
import luigi
import boto3
import numpy as np
import pandas as pd
from unittest import TestCase
from luigi import LuigiStatusCode
from tempfile import TemporaryDirectory
from luigi import ExternalTask, Parameter, Task
from pset_5.tasks.yelp_reviews import CleanedReviews, ByDecade, ByStars


"""
These first set of test are to ensure that we can 
1. Create a bucket in AWS
2. Access data from the bucket creaated in AWS
3. Download content for the created bucket, and store them in TmpDir Locally
Template for the first Test Case comes from: 
    https://medium.com/@l.peppoloni/how-to-mock-s3-services-in-python-tests-dd5851842946
"""


def create_tmp_bucket(keys, local_dir):
    s3 = boto3.resource("s3")
    local_paths = [os.path.join(local_dir, key) for key in keys]
    for key, local_path in zip(keys, local_paths):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.Bucket(name="pset-5").download_file(key, local_path)


class TestBotoConnection(TestCase):
    def test_download_csv_files(self):
        s3 = boto3.resource("s3")
        objects = s3.Bucket(name="pset-5").objects.filter(Prefix="yelp_data/")
        keys = [obj.key for obj in objects if obj.key.endswith(".csv")]
        desired_result = [
            "yelp_data/yelp_subset_0.csv",
            "yelp_data/yelp_subset_1.csv",
            "yelp_data/yelp_subset_10.csv",
            "yelp_data/yelp_subset_11.csv",
            "yelp_data/yelp_subset_12.csv",
            "yelp_data/yelp_subset_13.csv",
            "yelp_data/yelp_subset_14.csv",
            "yelp_data/yelp_subset_15.csv",
            "yelp_data/yelp_subset_16.csv",
            "yelp_data/yelp_subset_17.csv",
            "yelp_data/yelp_subset_18.csv",
            "yelp_data/yelp_subset_19.csv",
            "yelp_data/yelp_subset_2.csv",
            "yelp_data/yelp_subset_3.csv",
            "yelp_data/yelp_subset_4.csv",
            "yelp_data/yelp_subset_5.csv",
            "yelp_data/yelp_subset_6.csv",
            "yelp_data/yelp_subset_7.csv",
            "yelp_data/yelp_subset_8.csv",
            "yelp_data/yelp_subset_9.csv",
        ]
        self.assertEqual(sorted(keys)[0], sorted(desired_result)[0])
        self.assertEqual(sorted(keys)[5], sorted(desired_result)[5])
        self.assertEqual(sorted(keys)[10], sorted(desired_result)[10])
        self.assertEqual(sorted(keys)[-1], sorted(desired_result)[-1])

        with TemporaryDirectory() as tmpdir:
            create_tmp_bucket(keys, tmpdir)
            mock_folder_local_path = os.path.join(tmpdir, "yelp_data/")
            self.assertTrue(os.path.isdir(mock_folder_local_path))
            result = os.listdir(mock_folder_local_path)
            print("RESULT", result)
            desired_result = [
                "yelp_subset_0.csv",
                "yelp_subset_1.csv",
                "yelp_subset_10.csv",
                "yelp_subset_11.csv",
                "yelp_subset_12.csv",
                "yelp_subset_13.csv",
                "yelp_subset_14.csv",
                "yelp_subset_15.csv",
                "yelp_subset_16.csv",
                "yelp_subset_17.csv",
                "yelp_subset_18.csv",
                "yelp_subset_19.csv",
                "yelp_subset_2.csv",
                "yelp_subset_3.csv",
                "yelp_subset_4.csv",
                "yelp_subset_5.csv",
                "yelp_subset_6.csv",
                "yelp_subset_7.csv",
                "yelp_subset_8.csv",
                "yelp_subset_9.csv",
            ]
            self.assertEqual(sorted(result), sorted(desired_result))


class TestPipeline(TestCase):
    def testCleanedReviewsBuild(self):
        build_status = luigi.build(
            [CleanedReviews()], local_scheduler=True, detailed_summary=True
        ).status
        self.assertEqual(build_status, LuigiStatusCode.SUCCESS)

    def testCleanedReviewsFileExists(self):
        build_status = luigi.build(
            [CleanedReviews()], local_scheduler=True, detailed_summary=True
        ).status
        self.assertTrue(os.path.exists("./data/CleanedReviews-files/"))
        self.assertTrue(os.path.exists("./data/CleanedReviews-files/part.0.parquet"))

    def testByDecadeBuild(self):
        build_status = luigi.build(
            [ByDecade()], local_scheduler=True, detailed_summary=True
        ).status
        self.assertEqual(build_status, LuigiStatusCode.SUCCESS)

    def testByDecadeFileExists(self):
        build_status = luigi.build(
            [CleanedReviews()], local_scheduler=True, detailed_summary=True
        ).status
        self.assertTrue(os.path.exists("./data/ByDecade/"))
        self.assertTrue(os.path.exists("./data/ByDecade/part.0.parquet"))

    def testByStarsBuild(self):
        build_status = luigi.build(
            [ByStars()], local_scheduler=True, detailed_summary=True
        ).status
        self.assertEqual(build_status, LuigiStatusCode.SUCCESS)

    def testByStarsFileExists(self):
        build_status = luigi.build(
            [CleanedReviews()], local_scheduler=True, detailed_summary=True
        ).status
        self.assertTrue(os.path.exists("./data/ByStars/"))
        self.assertTrue(os.path.exists("./data/ByStars/part.0.parquet"))
    
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
  
    def testByStarsValues(self):
        df_byStars = pd.read_parquet("./data/ByStars/part.0.parquet")
        self.assertEqual(df_byStars["review_len"][1], 772)
        self.assertEqual(df_byStars["review_len"][2], 751)
        self.assertEqual(df_byStars["review_len"][3], 712)
        self.assertEqual(df_byStars["review_len"][4], 605)
        self.assertEqual(df_byStars["review_len"][5], 488)

    def testByDecadeValues(self):
        df_byDecade = pd.read_parquet("./data/ByDecade/part.0.parquet")
        self.assertEqual(df_byDecade["review_len"][2000], 763)
        self.assertEqual(df_byDecade["review_len"][2010], 595)
