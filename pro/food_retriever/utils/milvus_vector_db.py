import pandas as pd
import ast
import gc
import time
import pymilvus
import json
from pymilvus import MilvusClient
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    AnnSearchRequest, RRFRanker, WeightedRanker,
)
import os
from FlagEmbedding import BGEM3FlagModel
import numpy as np


def sample_list(input_list, percentage):
    # Ensure the percentage is between 0 and 100
    percentage = max(0, min(100, percentage))

    # Calculate the number of elements to sample
    sample_size = int(len(input_list) * (percentage / 100))

    # Use numpy to randomly sample without replacement
    sampled_indices = np.random.choice(len(input_list), size=sample_size, replace=False)

    # Return the sampled elements
    return [input_list[i] for i in sampled_indices]


sample_percentage = 25

client = MilvusClient("./dev/db/milvus_recipe.db")
print("Haciendo conexión con milvus vector data base...")

if client.has_collection('recipe_collection'):
    print("Cargando collection...")
    client.load_collection("recipe_collection")

else:
    start = time.time()
    print("Generando collection...")
    print("Puede tardar unos minutos")

    path_data = '/home/erikmn/PycharmProjects/SmartFridge/dev/db/clean_food_recipes.csv'
    dataset_df = pd.read_csv(path_data, converters={'text_embedding': ast.literal_eval})
    dataset_df = dataset_df.dropna()
    gc.collect()

    #dataset_df['text_embedding'] = dataset_df['text_embedding'].apply(ast.literal_eval)
    dataset_df = dataset_df[['id', 'name', 'minutes', 'steps', 'n_steps', 'ingredients', 'n_ingredients', 'description', 'text_1', 'text_embedding']]
    gc.collect()

    print("Columns:", dataset_df.columns)
    print("Dims:", dataset_df.shape)

    vector = dataset_df.values

    data = [ {"id": ivector[0], "vector":ivector[9], "text":ivector[8], "name": ivector[1], "time": ivector[2],
              "steps": ivector[3], "n_steps":ivector[4], "ingredients":ivector[5], "n_ingredients":ivector[6],
              "description":ivector[7]} for ivector in vector]

    end = time.time()
    print(end - start)

    sample_data = sample_list(input_list=data, percentage=sample_percentage)
    print(f"{sample_percentage}% sample of the list: {len(sample_data)} out of {len(data)}")


    start = time.time()
    client.create_collection(
        collection_name="recipe_collection",
        dimension=768  # The vectors we will use in this demo has 384 dimensions
    )

    res = client.insert(
        collection_name="recipe_collection",
        data=sample_data
    )
    end = time.time()
    print(end - start)

    res = json.dumps(res, indent=4)
    print(res)

    # 5. View Collections
    res = client.describe_collection(
        collection_name="recipe_collection"
    )
    print(res[0])
