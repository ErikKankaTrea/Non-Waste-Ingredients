import re
import json
from pymilvus import MilvusClient
from pymilvus.model.reranker import BGERerankFunction
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

class VectorSearchDB:
    def __init__(self, client_path_name, collection_name, amplitud = 300):
        self.client_path_name = client_path_name
        self.collection_name = collection_name
        self.client = MilvusClient(self.client_path_name)
        self.search_params = {"metric_type": "COSINE", "params": {"nlist": amplitud}}
        self.model = BGEM3FlagModel('BAAI/bge-base-en', use_fp16=True) #BAAI/bge-large-en BAAI/bge-m3
        self.bge_rf = BGERerankFunction(device='cpu')

    def load_collection(self):
        try:
            print("Cargando collection...")
            self.client.load_collection(self.collection_name)
        except ValueError as e:
            print("None collection with that name exist. Try to do list collections." )

    def create_milvus_filter(self, n_steps: list = None, time: int = None):
        if n_steps is not None and time is not None:
            filter_to_db = f'n_steps >= {n_steps[0]} and n_steps <= {n_steps[1]} and time<={time}'
        elif n_steps is None and time is not None:
            filter_to_db = f'time<="{time}"'
        elif n_steps is not None and time is None:
            filter_to_db = f'n_steps >= {n_steps[0]} and n_steps <= {n_steps[1]}'
        else:
           filter_to_db = f''
        return filter_to_db


    def make_dict_inputs(self, INPUT_INGREDIENTS, input1, input2, input3):
        tags = ', '.join(input1)
        n_steps = list(input2)
        time = int(input3)
        ingredients = ', '.join(INPUT_INGREDIENTS)
        return {'n_steps':n_steps, 'time':time, 'ingredients':ingredients, 'tags':tags}


    def get_embedding(self, text):
       """Generates vector embeddings for the given text."""
       embedding = self.model.encode(text)
       return embedding['dense_vecs'].tolist()


    def make_query(self, ingredients, tags):
        if tags is None and ingredients is not None:
            return (f"This dish is made of the following ingredients:\n"
                    f"{ingredients}\n")
        elif tags is not None and ingredients is not None:
            return (f"This dish is made of the following ingredients:\n"
                    f"{ingredients}\n"
                    f"The following tags describe the recipe:\n"
                    f"{tags}")
        elif tags is not None and ingredients is None:
            return (f"The following tags describe the recipe:\n"
                    f"{tags}")
        else:
            return f""

    def query_and_search(self, input_dict: dict, use_reranker = True):
        milvus_filter = self.create_milvus_filter(input_dict['n_steps'], input_dict['time'])
        prompt_to_embedding = self.make_query(input_dict['ingredients'], input_dict['tags'])
        query_embedding = [self.get_embedding(prompt_to_embedding)]
        res = self.client.search(
            collection_name="recipe_collection",
            data=query_embedding,
            filter=milvus_filter,
            limit=5,
            output_fields=["n_steps", "time", "description", "ingredients", "steps", "text"],
            search_params=self.search_params)
        if use_reranker:
            result_texts = [re.sub("\[|\]|'", "", hit['entity']["text"]) for hit in res[0]]
            res_ranked = self.bge_rf(prompt_to_embedding, result_texts, top_k=5)
            # rerank the results using BGE CrossEncoder model
            res = [hit for hit in res[0] if res_ranked[0].text in hit['entity']["text"]]
        return res

