import os
import re

# 1. Dataset Loading
from datasets import load_dataset
import pandas as pd
import numpy as np
import time

path_data = '/home/erikmn/PycharmProjects/mongo_DB_retriever/data/food_recipes.csv'
dataset_df = pd.read_csv(path_data)
dataset_df.head(5)
dataset_df.columns

from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel('BAAI/bge-base-en', use_fp16=True) #BAAI/bge-large-en BAAI/bge-m3

def get_embedding(text):
   """Generates vector embeddings for the given text."""
   embedding = model.encode(text)
   return embedding['dense_vecs'].tolist()

def create_text_1(col1, col2):
   """Generates text column to be passed for the embedding model."""
   return  ("This dish is made of the following ingredients:\n" +
               re.sub("\[|\]|'", "", col1) + "\n" +
            "The following tags describe the recipe:\n" +
            re.sub("\[|\]|'", "", col2))

def create_text_2(col1, col2, col3):
   """Generates text column to be passed for the embedding model."""
   return  (str(col3) + "\n" +
            "The dish is made of the following ingredients:\n" +
            re.sub("\[|\]|'", "", col1) + "\n" +
            "The following tags describe the recipe:\n" +
            re.sub("\[|\]|'", "", col2))

def extract_time(string):
   """Extract time making dish."""
   pattern = r"'([^']*(?:-or-less|-or-more)[^']*)'"
   match = re.search(pattern, string)
   return match.group(1) if match else None

# Minutes metadata:
dataset_df['time'] = dataset_df['tags'].apply(lambda x: extract_time(x))
dataset_df["text_1"] = dataset_df.apply(lambda x: create_text_1(x.ingredients, x.tags), axis=1)
#dataset_df["text_2"] = dataset_df.apply(lambda x: create_text_2(x.ingredients, x.tags, x.description), axis=1)

# Creates embeddings and stores them as a new field
#df = dataset_df.iloc[0:5]
start_time = time.time()
dataset_df["text_embedding"] = dataset_df["text_1"].apply(get_embedding)
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

dataset_df.to_csv('/home/erikmn/PycharmProjects/mongo_DB_retriever/food_recipes_clean.csv')
