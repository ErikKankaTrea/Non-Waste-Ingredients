# main.py
import datetime
import os
import json


class FoodLookUp():

    def __init__(self, food_db):
        #self.db_path_file = db_path_file
        #with open(self.db_path_file) as db_file:
        #    food_db = json.load(db_file)
        self.food_db = food_db
        self.session_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    def db_checker(self, value, look_up_by):
        if look_up_by == 'name':
            return any([value in i_element['name'] for i_element in self.food_db['food_db']])
        else:
            return any([value in i_element['ean_id'] for i_element in self.food_db['food_db']])


    def add_to_db(self, value, look_up_by):
        if self.db_checker(value, look_up_by):
            idx_value = [i for i, x in enumerate(self.food_db['food_db']) if value in x[look_up_by]]
            retrieved_food = self.food_db['food_db'][idx_value[0]]
            retrieved_food['date'] = self.session_date
            retrieved_food['status'] = 'buen estado'
            retrieved_food['days_at_home'] = 0
            #self.food_db['food_db'].append(retrieved_food) # Si se activa se guarda automaticamente si existe una compra
        else:
            retrieved_food = RetrieverByCode(value)
            #if 'OK' in retriever_code.process_code():
            #    retrieved_food = retriever_code.get_info_from_OF()
            #    if retriever_code.set_status(retrieved_food):
            #        print(retrieved_food)
            #    else:
            #        print("añadelo de otra manera")
            #else:
            #    "No encontrado"
        return retrieved_food


    def sync_db(self, limit):
        for i_element in self.food_db['food_db']:
            diff_days = datetime.datetime.strptime(self.session_date, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(
                i_element['date'], '%Y-%m-%d %H:%M:%S')
            total_days = diff_days.days
            i_element['days_at_home'] = total_days
            i_element['estado'] = 'buen estado' if total_days < limit else 'mal estado'

