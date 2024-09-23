import requests
import datetime
import re
import os
import json
from dotenv import load_dotenv
load_dotenv()

class RetrieverByCode:
    def __init__(self, input_code):
        self.input_code = input_code
        self.session_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.edam_api = {'homepage': 'https://developer.edamam.com/', 'url': 'https://api.edamam.com/api/nutrition-data', 'auth': {'app_id': os.environ["APP_ID"], 'app_key': os.environ["EDAMAM_API_KEY"]}, 'query_str': {'ingr': '', 'nutrition-type': 'logging'}}
        self.headers = {"Accept": "application/json", }
        self.api_url = self.edam_api['url']
        self.api_auth = self.edam_api['auth']
        self.url_of = 'https://world.openfoodfacts.org/api/v2/product/{}.json'


    def process_code(self):
        if len(self.input_code) > 0:
            return "STATUS:OK"
        else:
            return "STATUS:BAD"



    def get_elements_from_OF(self, dict_json, type):
        if 'name' in type:
            try:
                value = dict_json['product'][type]
            except:
                value = ''
        else:
            try:
                value = dict_json['product']['nutriments'][type]
            except:
                value = -1.0
        return value


    def get_elements_from_edamam(self, dict_json, type):
        try:
            value = dict_json['totalNutrients'][type]['quantity']
        except:
            value = -1.0
        return value


    def add_header(self, res):
        # Include cookie for every request
        res.headers['Access-Control-Allow-Origin'] = True
        # Prevent the client from caching the response
        if 'Cache-Control' not in res.headers:
            res.headers['Cache-Control'] = 'public, no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
            res.headers['Pragma'] = 'no-cache'
            res.headers['Expires'] = '-1'
        return res

    def get_info_from_OF(self):
        url = self.url_of.format(self.input_code)
        res = requests.get(url)
        res = self.add_header(res)
        if res.status_code == 200:
            res_json = res.json()
            return {
                    'date': self.session_date,
                    'ean_id': self.input_code,
                    'status': 'buen estado',
                    'product_name': self.get_elements_from_OF(dict_json=res_json, type='product_name'),
                    'generic_name': self.get_elements_from_OF(dict_json=res_json, type='generic_name'),
                    'calories': self.get_elements_from_OF(dict_json=res_json, type='energy-kcal'),
                    'fat': self.get_elements_from_OF(dict_json=res_json, type='fat'),
                    'carbs': self.get_elements_from_OF(dict_json=res_json, type='carbohydrates'),
                    'protein': self.get_elements_from_OF(dict_json= res_json, type='proteins'),
                    #'sugar': self.get_elements_from_OF(dict_json= res_json, type='sugars'),
                    'fiber':-1,
                    #'lan': self.get_elements_from_OF(dict_json= res_json, type='languages_hierarchy'),
                    'days_at_home': 0
                    }
        else:
            return {
                'date': self.session_date,
                'ean_id': self.input_code,
                'status': '',
                'product_name': '',
                'generic_name': '',
                'calories': -1,
                'fat': -1,
                'carbs': -1,
                'protein': -1,
                #'sugar': -1,
                'fiber': -1,
                #'lan': None,
                'days_at_home': -1
            }



    def get_info_from_edamam(self, label):
        headers = {"Accept": "application/json", }
        api_url = self.edam_api['url']
        api_auth = self.edam_api['auth']
        query_str = self.edam_api['query_str']
        query_str['ingr'] = label
        input_params = {}
        input_params.update(api_auth)
        params = query_str
        input_params.update(params)

        response = requests.get(
            api_url,
            params=input_params,
            headers=headers)

        response = self.add_header(response)

        if response.status_code == 200:
            res_json = response.json()
            return {
                    'date': self.session_date,
                    'ean_id': "-1",
                    'status': 'buen estado',
                    'product_name': self.input_code,
                    'generic_name': self.input_code,
                    'calories': self.get_elements_from_edamam(dict_json=res_json, type='ENERC_KCAL'),
                    'fat': self.get_elements_from_edamam(dict_json=res_json, type='FAT'),
                    'carbs': self.get_elements_from_edamam(dict_json=res_json, type='CHOCDF'),
                    'protein': self.get_elements_from_edamam(dict_json= res_json, type='PROCNT'),
                    #'sugar': self.get_elements_from_edamam(dict_json= res_json, type='SUGAR'),
                    'fiber': self.get_elements_from_edamam(dict_json= res_json, type='FIBTG'),
                    #'lan': self.get_elements_from(dict_json= res_json, type='languages_hierarchy'),
                    'days_at_home': 0
                    }
        else:
            return {
                'date': self.session_date,
                'ean_id': -1,
                'status': '',
                'product_name': label,
                'generic_name': label,
                'calories': -1,
                'fat': -1,
                'carbs': -1,
                'protein': -1,
                #'sugar': -1,
                'fiber': -1,
                #'lan': None,
                'days_at_home': -1
            }


    def set_status(self, of_result):
        return True if (of_result['product_name'] != None) and (of_result['product_name'] != '') else False




class FoodLookUp(RetrieverByCode):

    def __init__(self, food_db):
        #self.db_path_file = db_path_file
        #with open(self.db_path_file) as db_file:
        #    food_db = json.load(db_file)
        self.food_db = food_db
        self.session_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    def db_checker(self, value, look_up_by):
        return any([value in i_element[look_up_by] for i_element in self.food_db['food_db']])


    def add_to_db_with_of(self, value, look_up_by):
        if self.db_checker(value, look_up_by):
            idx_value = [i for i, x in enumerate(self.food_db['food_db']) if value in x[look_up_by]]
            retrieved_food = self.food_db['food_db'][idx_value[0]]
            retrieved_food['date'] = self.session_date
            retrieved_food['status'] = 'buen estado'
            retrieved_food['days_at_home'] = 0
            self.food_db['food_db'].append(retrieved_food)
        else:
            retriever_code = RetrieverByCode(value)
            assert 'OK' in retriever_code.process_code(), 'Código sin longitud!'
            retrieved_food = retriever_code.get_info_from_OF()
        return retrieved_food



    def add_to_db_with_edamam(self, value, look_up_by):
        if self.db_checker(value, look_up_by):
            print("Estamos dentro del check en la BD")
            idx_value = [i for i, x in enumerate(self.food_db['food_db']) if value in x[look_up_by]]
            retrieved_food = self.food_db['food_db'][idx_value[0]]
            retrieved_food['date'] = self.session_date
            retrieved_food['status'] = 'buen estado'
            retrieved_food['days_at_home'] = 0
            self.food_db['food_db'].append(retrieved_food)
        else:
            print("Estamos dentro del check EN LA API")
            retriever_code = RetrieverByCode(value)
            print("Instanced CREATED!")
            assert 'OK' in retriever_code.process_code(), 'Código sin longitud!'
            retrieved_food = retriever_code.get_info_from_edamam(value)
        return retrieved_food


    def categorize_nutrient(self, amount: float, nutrient_type: str) -> str:
        nutrient_thresholds = {
            'calories': (40, 100),
            'fat': (3, 17.5),
            'carbs': (5, 22.5),
            'protein': (5, 20),
            'sugar': (5, 22.5),
            'fiber': (2, 6)
        }

        if nutrient_type.lower() not in nutrient_thresholds:
            raise ValueError(f"Unknown nutrient type: {nutrient_type}")

        lower, upper = nutrient_thresholds[nutrient_type.lower()]
        if amount == -1:
            return "none"
        elif amount < lower:
            return "low"
        elif amount <= upper:
            return "medium"
        else:
            return "high"


    def sync_db(self, limit):
        for i_element in self.food_db['food_db']:
            diff_days = datetime.datetime.strptime(self.session_date, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(
                i_element['date'], '%Y-%m-%d %H:%M:%S')
            total_days = diff_days.days
            i_element['days_at_home'] = total_days
            i_element['status'] = 'buen estado' if total_days < limit else 'mal estado'
            i_element['in_calories'] = self.categorize_nutrient(i_element['calories'], 'calories')
            i_element['in_fat'] = self.categorize_nutrient(i_element['fat'], 'fat')
            i_element['in_carbs'] = self.categorize_nutrient(i_element['carbs'], 'carbs')
            i_element['in_protein'] = self.categorize_nutrient(i_element['protein'], 'protein')
            #i_element['in_sugar'] = self.categorize_nutrient(i_element['sugar'], 'sugar')
            i_element['in_fiber'] = self.categorize_nutrient(i_element['fiber'], 'fiber')