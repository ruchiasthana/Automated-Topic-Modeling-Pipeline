import numpy as np
import pandas as pd
from os import path
from PIL import Image
import functools
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

class CreateWordCloud(object):
    def __init__(self, text_path, field_name, output_path):
        self.output_path = output_path
        self.text_path = text_path
        df = pd.read_csv(text_path)
        df = df.head(50000)
        self.text = functools.reduce(lambda x,y: str(x)+ " " + str(y), df[field_name])
    
    def create_wordcloud(self):
        wordcloud = WordCloud().generate(str(self.text))
        wordcloud.to_file(self.output_path)
    
    def main(self):
        self.create_wordcloud()