from pipes import stepkinds

import streamlit as st
import datetime
import json
import os
import sys
from ultralytics import YOLOv10

import cv2
import supervision as sv

import logging
import queue
from pathlib import Path
from typing import List, NamedTuple

import av
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client

import ast
import re
from datetime import datetime
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

from typing_extensions import TypedDict
from typing import List
import os, re, time
from colorama import Fore, Style
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from FlagEmbedding import BGEM3FlagModel


# Define yout paths
root_dir = '/home/erikmn/PycharmProjects/SmartFridge/dev'
project_dir = '/home/erikmn/PycharmProjects/SmartFridge/dev/food_retriever'
json_file_path = '/home/erikmn/PycharmProjects/SmartFridge/dev/db/food_db.json'
buckup_json_file_path = '/home/erikmn/PycharmProjects/SmartFridge/dev/db/buckup_food_db.json'
image_file_path = '/home/erikmn/PycharmProjects/SmartFridge/dev/images/images/fridge.png'
music_path = '/home/erikmn/PycharmProjects/SmartFridge/dev/food_retriever/assets/jazz-bossa-nova-163669.mp3'
yolo_model_path = '/home/erikmn/PycharmProjects/SmartFridge/dev/food_retriever/yolo_weights/v3_w_yolov10_n/best.pt'
milvus_db_path = './dev/db/milvus_recipe.db'
collection_name = 'recipe_collection'

sys.path.append(project_dir)

from yolov10.ultralytics.utils.callbacks import default_callbacks
from utils.food_retriever_of import RetrieverByCode, FoodLookUp
from utils.multi_agent import AgentState, Agents, Nodes, Workflow
from utils.milvus_search_engine import VectorSearchDB
from utils.dish_with_llm import CookAssistant
from utils import multi_agent


MAX_REVISIONS = 1
STATUS_LIMIT = 8
api_key=os.environ["GROQ_API_KEY"]
model_name="llama3-8b-8192"
sp_CLASSES = ['aceituna', 'aguacate', 'ajo', 'albaricoque', 'apio', 'aquarius', 'berenjena', 'bolsa de ajos', 'bolsa de patatas', 'brocoli', 'calabacin', 'calamar', 'carpaccio', 'cebolla', 'cereza', 'cerveza', 'cerveza estrella', 'champinon laminado', 'chorizo', 'couscous', 'croissant', 'escalivada', 'espinacas', 'fideos', 'flan', 'fresa', 'gazpacho', 'hortalizas', 'huevo', 'jamon', 'judias verdes', 'kiwi', 'langostino', 'lechuga', 'limon', 'longaniza', 'manzana', 'melocoton', 'menta', 'naranja', 'pan', 'patata', 'pavo', 'pechuga de pollo', 'pera', 'perejil', 'pimiento rojo', 'pimiento verde', 'pina', 'pistacho', 'pizza', 'platano', 'poke salmon', 'puerro', 'queso', 'queso de loncha', 'racimo uvas', 'salchichas', 'salmon', 'salsa de tomate', 'sandia', 'shushi', 'solomillo decerdo', 'special K', 'surimi', 'ternera', 'tomate', 'vino blanco', 'yogourt griego', 'zanahoria']
en_CLASSES = ['olive', 'avocado', 'garlic', 'apricot', 'celery', 'aquarius', 'eggplant', 'bag of garlic', 'bag of potatoes', 'broccoli', 'zucchini', 'squid', 'carpaccio', 'onion', 'cherry', 'beer', 'star beer', 'sliced ​​mushroom', 'chorizo', 'couscous', 'croissant', 'escalivada', 'spinach', 'noodles', 'flan', 'strawberry', 'gazpacho', 'vegetables', 'egg', 'ham', 'green beans', 'kiwi', 'shrimp', 'lettuce', 'lemon', 'sausage', 'apple', 'peach', 'mint', 'orange', 'bread', 'potato', 'turkey', 'chicken breast', 'pear', 'parsley', 'red pepper', 'green pepper', 'pineapple', 'pistachio', 'pizza', 'banana', 'poke salmon', 'leek', 'cheese', 'sliced cheese', 'bunch of grapes', 'sausages', 'salmon', 'tomato sauce', 'watermelon', 'shushi', 'pork tenderloin', 'special K', 'surimi', 'beef', 'tomato', 'white wine', 'Greek yogurt', 'carrot']
DICT_CLASSES = {sp_CLASSES[i]: en_CLASSES[i] for i in range(len(sp_CLASSES))}
TAG_LIST = [None, 'chocolate-chip-cookies', 'frozen-desserts', 'berries', 'presentation', 'lentils', 'food-processor-blender', 'free-of-something', 'easter', 'strawberries', 'spinach', 'libyan', 'low-saturated-fat', 'spicy', 'pork', 'pizza', 'goose', 'mothers-day', 'canning', 'ice-cream', 'valentines-day', 'lebanese', 'soups-stews', 'marinara-sauce', 'beef-ribs', 'heirloom-historical', 'vietnamese', 'halloween-cupcakes', 'meatballs', 'chicken-stews', 'omelets-and-frittatas', 'a1-sauce', 'tomatoes', 'mardi-gras-carnival', 'turkey-breasts', 'candy', 'side-dishes-beans', 'turkish', 'meat', 'fall', 'romantic', 'penne', 'nigerian', 'whole-duck', 'stuffings-dressings', 'ground-beef', 'superbowl', 'guatemalan', 'eggs', 'diabetic', 'somalian', 'bear', 'veal', 'breakfast-potatoes', 'french', 'mediterranean' ,'hidden-valley-ranch', 'norwegian', 'thanksgiving', 'gelatin', 'angolan', 'eggs-dairy', 'beans', 'dietary', 'april-fools-day', 'lasagna', 'chilean', 'no-shell-fish', 'shakes', 'to-go', 'chowders', 'swedish', 'pancakes-and-waffles', 'rabbit', 'raspberries', 'mixer', 'main-dish-pasta', 'finger-food', 'avocado', 'chicken', 'low-carb', 'whitefish', 'quiche', 'mushroom-soup', 'ramadan', 'bass', 'snacks-sweet', 'italian', 'japanese', 'mushrooms', 'southwestern-united-states', 'beef-barley-soup', 'citrus', 'new-zealand', 'lamb-sheep-main-dish', 'vegetarian', 'served-hot', 'microwave', 'cheese', 'yeast', 'chicken-stew', 'bread-machine', 'comfort-food', 'poultry', 'passover', 'manicotti', 'hunan', 'finnish', 'wings', 'pheasant', 'bread-pudding', 'halibut', 'ham', 'salmon', 'icelandic', '15-minutes-or-less', 'for-1-or-2', 'simply-potatoes2', '3-steps-or-less', 'served-cold', 'elk', 'brunch', 'rolled-cookies', 'namibian', 'shrimp', 'lunch', 'czech', 'fudge', 'soy-tofu', 'vegan', '1-day-or-more', 'saltwater-fish', 'birthday', 'bacon', 'salad-dressings', 'college', 'pot-pie', 'condiments-etc', 'independence-day', 'drop-cookies', 'copycat', 'chinese', 'pineapple', 'sudanese', 'curries', 'cinco-de-mayo', 'leftovers', 'smoothies', 'swiss', 'broil', 'central-american', 'greek', 'pot-roast', 'grilling', 'catfish', 'jellies', 'holiday-event', 'low-in-something', 'preparation', 'main-dish', 'freezer', 'dinner-party', 'sweet', 'snacks', 'tempeh', 'chick-peas-garbanzos', 'labor-day', 'moroccan', 'main-dish-beef', 'lemon', 'deep-fry', 'asparagus', 'pies', 'english', 'summer', 'memorial-day', 'pitted-fruit', 'quick-breads', 'carrots', 'stove-top', 'squid', 'american', 'main-dish-pork', 'long-grain-rice', 'pasta', 'nuts', 'chicken-crock-pot', 'squash', 'beef', 'irish-st-patricks-day', 'oranges', 'indian', 'eggs-breakfast', 'short-grain-rice', 'pasta-shells', 'british-columbian', 'soul', 'roast', 'celebrity', 'jewish-ashkenazi', 'roast-beef', 'mexican', 'oaxacan', 'pickeral', 'egg-free', 'quebec', 'lamb-sheep', 'baked-beans', 'orange-roughy', 'ontario', 'belgian', 'biscotti', 'collard-greens', 'papaya', 'pork-crock-pot', 'pork-loin', 'crab', 'onions', 'lobster', 'ecuadorean', 'tarts', 'st-patricks-day', 'simply-potatoes', 'peanut-butter', 'cambodian', 'super-bowl', 'potluck', 'beef-sauces', 'main-dish-seafood', 'asian', 'northeastern-united-states', 'malaysian', 'hand-formed-cookies', 'nepalese', 'welsh', 'clams', 'sugar-cookies', 'ragu-recipe-contest', 'seafood', 'amish-mennonite', 'californian', 'pennsylvania-dutch', 'salsas', 'flat-shapes', 'midwestern', 'pumpkin', 'tex-mex', 'congolese', 'oamc-freezer-make-ahead', 'reynolds-wrap', 'african', 'jams-and-preserves', 'nut-free', 'pork-chops', 'winter', 'chutneys', 'honduran', 'easy', 'from-scratch', 'whole-chicken', 'artichoke', 'stir-fry', 'iranian-persian', 'served-hot-new-years', 'kid-friendly', 'dips-summer', 'chicken-livers', 'beef-crock-pot', 'snacks-kid-friendly', 'scones', 'stews-poultry', 'macaroni-and-cheese', 'roast-beef-main-dish', 'sauces', 'unprocessed-freezer', 'camping', 'chicken-thighs-legs', 'cooking-mixes', 'trout', 'plums', 'wild-game', 'weeknight', 'colombian', 'halloween-cakes', 'beef-sausage', 'filipino', 'cupcakes', 'for-large-groups', 'punch', 'austrian', 'desserts-easy', 'heirloom-historical-recipes', 'grains', 'corn', 'low-cholesterol', 'octopus', 'laotian', 'rolls-biscuits', 'halloween', 'ethiopian', 'venezuelan', 'ham-and-bean-soup', 'cabbage', '5-ingredients-or-less', 'middle-eastern-main-dish', 'cod', 'water-bath', 'bananas', 'for-large-groups-holiday-event', 'zucchini', 'blueberries', 'wedding', 'fish', 'meatloaf', 'barbecue', 'danish', 'new-years', 'desserts-fruit', 'beginner-cook', 'very-low-carbs', 'cuisine', 'scandinavian', 'sweet-sauces', 'eggplant', 'number-of-servings', 'mango', 'canadian', 'tuna', 'melons', 'turkey-burgers', 'pacific-northwest', 'micro-melanesia', 'breakfast-eggs', 'cherries', 'peaches', 'spaghetti', 'taste-mood', 'bean-soup', 'gifts', 'pasta-elbow-macaroni', 'spring', 'steam', 'cauliflower', 'hungarian', 'lime', 'dutch', 'oatmeal', 'mashed-potatoes', 'beef-kidney', 'course', 'main-ingredient', '60-minutes-or-less', 'kwanzaa', 'breakfast-casseroles', 'beijing', 'lactose', 'technique', 'thai', 'stews', 'vegetables', 'seasonal', 'jewish-sephardi', 'southern-united-states', 'gumbo', 'oysters', 'pears', 'portuguese', 'less_thansql:name_topics_of_recipegreater_than', 'breads', 'polish', 'side-dishes', 'chard', 'small-appliance', 'chicken-breasts', 'non-alcoholic', 'cookies-and-brownies', 'casseroles', 'baja', 'australian', 'white-rice', 'german', 'duck', 'freshwater-fish', 'south-african', 'tilapia', 'healthy-2', 'cranberry-sauce', 'oven', 'crock-pot-slow-cooker', 'north-american', 'breakfast', 'costa-rican', 'garnishes', 'creole', 'brazilian', 'coffee-cakes', 'infant-baby-friendly', 'cuban', 'mahi-mahi', 'savory', 'salads', 'cocktails', 'appetizers', 'cakes', 'iraqi', 'ravioli-tortellini', 'deer', 'brewing', 'pasta-rice-and-grains', 'korean', 'greens', 'fruit', 'south-west-pacific', 'elbow-macaroni', 'coconut', 'burgers', 'desserts', 'pressure-cooker', 'lettuces', 'indonesian', 'novelty', 'shellfish', 'marinades-and-rubs', 'pasta-salad', 'cake-fillings-and-frostings', 'chocolate', 'medium-grain-rice', 'roast-beef-comfort-food', 'pumpkin-bread', 'refrigerator', 'peppers', 'beans-side-dishes', 'brown-rice', 'low-protein', 'apples', 'clear-soups', 'high-in-something', 'scallops', 'green-yellow-beans', 'dairy-free', 'puerto-rican', 'pressure-canning', 'spanish', 'pork-loins', '4-hours-or-less', 'cantonese', 'sourdough', 'picnic', 'native-american', 'broccoli', 'potatoes', 'bok-choys', 'russian', 'egyptian', 'chinese-new-year', 'pasta-rice-and-grains-elbow-macaroni', 'shrimp-main-dish', 'muffins', 'dips', 'whole-turkey', 'middle-eastern', 'savory-sauces', 'halloween-cocktails', 'duck-breasts', 'rice', 'time-to-make', 'crawfish', 'sole-and-flounder', 'savory-pies', 'grapes', '30-minutes-or-less', 'fathers-day', 'bisques-cream-soups', 'high-calcium', 'peruvian', 'fillings-and-frostings-chocolate', 'beef-organ-meats', 'prepared-potatoes', 'pork-ribs', 'moose', 'stocks', 'crusts-pastry-dough-2', 'hawaiian', 'pies-and-tarts', 'kiwifruit', 'sandwiches', 'argentine', 'spreads', 'gluten-free', 'toddler-friendly', 'baking', 'main-dish-chicken', 'hanukkah', 'kosher', 'saudi-arabian', 'beef-liver', 'dips-lunch-snacks', 'lasagne', 'occasion', 'georgian', 'high-fiber', 'pork-sausage', 'szechuan', 'low-fat', 'pork-loins-roast', 'mongolian', 'palestinian', 'tropical-fruit', 'cobblers-and-crisps', 'yams-sweet-potatoes', 'scottish', 'granola-and-porridge', 'black-bean-soup', 'healthy', 'inexpensive', 'steak', 'polynesian', 'chili', 'low-calorie', 'rosh-hashana', 'rosh-hashanah', 'caribbean', 'pakistani', 'mussels', 'crock-pot-main-dish', 'puddings-and-mousses', 'cajun', 'high-protein', 'beverages', 'brown-bag', 'high-in-something-diabetic-friendly', 'one-dish-meal', 'dehydrator', 'spaghetti-sauce', 'herb-and-spice-mixes', 'turkey', 'black-beans', 'south-american', 'steaks', 'cheesecake', 'christmas', 'irish', 'brownies', 'veggie-burgers', 'european', 'no-cook', 'low-sodium', 'bar-cookies', 'smoker', 'quail']
llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.3)#llama-3.1-8b-instant
milvus_db_path = './dev/db/milvus_recipe.db'
collection_name = 'recipe_collection'

vs = VectorSearchDB(client_path_name=milvus_db_path,
                        collection_name=collection_name,
                        amplitud=300)
vs.load_collection()

st.set_page_config(page_title="Phood App", layout="wide")

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: list#np.ndarray


@st.cache_resource  # type: ignore
def generate_label_colors():
    return np.random.uniform(0, 255, size=(len(sp_CLASSES), 3))


COLORS = generate_label_colors()

def get_ice_servers():
    """Use Twilio's TURN server because Streamlit Community Cloud has changed
    its infrastructure and WebRTC connection cannot be established without TURN server now.  # noqa: E501
    We considered Open Relay Project (https://www.metered.ca/tools/openrelay/) too,
    but it is not stable and hardly works as some people reported like https://github.com/aiortc/aiortc/issues/832#issuecomment-1482420656  # noqa: E501
    See https://github.com/whitphx/streamlit-webrtc/issues/1213
    """

    # Ref: https://www.twilio.com/docs/stun-turn/api
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
        print("Twilio credentials are not set. Fallback to a free STUN server from Google.")# noqa: E501
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)

    try:
        token = client.tokens.create()
    except TwilioRestException as e:
        st.warning(
            f"Error occurred while accessing Twilio API. Fallback to a free STUN server from Google. ({e})"  # noqa: E501
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    return token.ice_servers



# NOTE: The callback will be called in another thread,
#       so use a queue here for thread-safety to pass the data
#       from inside to outside the callback.
# TODO: A general-purpose shared state object may be more useful.
result_queue: "queue.Queue[List[Detection]]" = queue.Queue()


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")

    # Run inference
    results = net.predict(image, imgsz=640, conf=score_threshold)
    res = results[0]

    detections = [
        Detection(
            class_id=int(detctn.boxes.cls[0]),
            label=sp_CLASSES[int(detctn.boxes.cls[0])],
            score=float(detctn.boxes.conf[0]),
            box=(detctn.boxes.xyxy.tolist()),
        )
        for detctn in res
    ]

    # Render bounding boxes and captions
    for detection in detections:
        caption = f"{detection.label}: {round(detection.score * 100, 2)}%"
        color = COLORS[detection.class_id]
        xmin, ymin, xmax, ymax = detection.box[0]

        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        cv2.putText(
            image,
            caption,
            (int(xmin), int(ymin) - 15 if int(ymin) - 15 > 15 else int(ymin) + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
    result_queue.put(detections)

    return av.VideoFrame.from_ndarray(image, format="bgr24")


# Initialize a list to store the dictionaries
def init_day():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def extract_label(detection_object):
    """Extracts the label value from a Detection object."""
    label_match = re.search(r"label='(.*?)'", str(detection_object))
    if label_match:
        return label_match.group(1)
    else:
        return None

# Function to read data from the JSON file
def read_json_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return []


# Function to write data to the JSON file
def write_json_file(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def filter_avalaible_ingredients(food_db):
    CURRENT_AVAILABLE_INGREDIENTS = []
    sorted_data = sorted(food_db['food_db'], key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d %H:%M:%S'), reverse=False)
    for i_food in sorted_data:
        if i_food['status'] == 'buen estado':
            CURRENT_AVAILABLE_INGREDIENTS.append(i_food['generic_name'])
    return CURRENT_AVAILABLE_INGREDIENTS



# Initialize session state with data from the JSON file and buckup
if 'data' not in st.session_state:
    #Read data food_db
    st.session_state['data'] = read_json_file(json_file_path)
    # Make backup:
    write_json_file(buckup_json_file_path, read_json_file(json_file_path))


# Session-specific caching
cache_key_1 = "object_detection_yolov10"
if cache_key_1 in st.session_state:
    net = st.session_state[cache_key_1]
else:
    net = YOLOv10(yolo_model_path)
    st.session_state[cache_key_1] = net


#Sidebar for audio
st.audio(music_path, format="audio/mpeg", loop=True, autoplay=True)

# Sidebar with navigation
section = st.sidebar.radio("Navigation", ["Boxes Comida", "Sincronizar Compra", "Batch cooking Chef", "Hazme un plato"])


if section == "Boxes Comida":

    # Create tabs within the Fridge Storage section
    tab1, tab2, tab3 = st.tabs(["Registro EAN", "Registro Visual", "Registro Manual"])


    with tab1:
        # Create two columns
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📊🍕 Registro por código de barras")
            with st.form(key='ean_form', clear_on_submit=True):
                value = st.text_input('EAN ID')
                submit_button_ean = st.form_submit_button(label='Submit to retriever API')

            if submit_button_ean:
                look_up = FoodLookUp(st.session_state['data'])
                api_data_retrieved = look_up.add_to_db_with_of(value, look_up_by='ean_id')
                product_data = {
                    'date': api_data_retrieved['date'],
                    'ean_id': api_data_retrieved['ean_id'],
                    'status': api_data_retrieved['status'],
                    'product_name': api_data_retrieved['product_name'],
                    'generic_name': api_data_retrieved['generic_name'],
                    'calories': api_data_retrieved['calories'],
                    'fat': api_data_retrieved['fat'],
                    'carbs': api_data_retrieved['carbs'],
                    'protein': api_data_retrieved['protein'],
                    'sugar': api_data_retrieved['sugar'],
                    'fiber': api_data_retrieved['fiber'],
                    'days_at_home': api_data_retrieved['days_at_home']
                }

                # Append the dictionary to the list in session state
                st.session_state['data']['food_db'].append(product_data)
                st.success("Product data submitted successfully!")
                # Write the updated list to the JSON file
                write_json_file(json_file_path, st.session_state['data'])

        with col2:
            st.subheader("🕸🕸 Datos Capturados.")
            if submit_button_ean:
                st.write(product_data)  # st.session_state['data']



    with tab2:

        col1, col2 = st.columns([2, 1])

        with col1:

            score_threshold = st.slider("Score threshold", 0.0, 1.0, 0.2, 0.05)
            webrtc_ctx = webrtc_streamer(
                key="object-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={
                    "iceServers": get_ice_servers(),
                    "iceTransportPolicy": "relay",
                },
                video_frame_callback=video_frame_callback,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            with st.form(key='label_form', clear_on_submit=True):
                submit_label_button = st.form_submit_button(label='Submit to retrieve from API')

            if st.checkbox("Show the detected labels", value=True):
                if webrtc_ctx.state.playing:
                    labels_placeholder = st.empty()
                    # NOTE: The video transformation with object detection and
                    # this loop displaying the result labels are running
                    # in different threads asynchronously.
                    # Then the rendered video frames and the labels displayed here
                    # are not strictly synchronized.
                    while True:
                        result = result_queue.get()
                        labels_placeholder.table(result)
                        labels = [i_res.label for i_res in result]

                        try:
                            labels = list(np.unique(labels))[0]
                        except:
                            labels = ''

                        if submit_label_button:
                            look_up_label = FoodLookUp(st.session_state['data'])
                            api_data_retrieved = look_up_label.add_to_db_with_edamam(DICT_CLASSES[labels], look_up_by='generic_name')

                            product_data = {
                                'status': 'buen estado',
                                'days_at_home': 0,
                                'date': init_day(),
                                'product_name': labels,
                                'generic_name': labels,
                                'calories': api_data_retrieved['calories'],
                                'fat': api_data_retrieved['fat'],
                                'carbs': api_data_retrieved['carbs'],
                                'protein': api_data_retrieved['protein'],
                                'sugar': api_data_retrieved['sugar'],
                                'fiber': api_data_retrieved['fiber'],
                                'ean_id': api_data_retrieved['ean_id'],
                            }

                            # Append the dictionary to the list in session state
                            st.session_state['data']['food_db'].append(product_data)
                            st.success("Product data submitted successfully!")
                            # Write the updated list to the JSON file
                            write_json_file(json_file_path, st.session_state['data'])
                            break

        with col2:
            st.subheader("🕸🕸 Datos Capturados.")
            if st.session_state['data']:
                st.write(st.session_state['data'])

        st.markdown(
            "Muestra el producto a la camara y haz click en el botón 'Submit to retrieve from API'."  
            "Se intentará coger los datos nutritivos a traves de EDAMAM y PHO DB."
        )


    with tab3:
        # Create two columns
        col1, col2 = st.columns([1, 1])

        with col1:
            # Streamlit form
            st.subheader("✍️🍕 Registro manual")

            with st.form(key='product_form', clear_on_submit=True):
                product_name = st.text_input('Product Name')
                product = st.text_input('General Name')
                calories = st.number_input('Calories', min_value=-1)
                fat = st.number_input('Fat (g)', min_value=-1.0, format="%.2f")
                carbs = st.number_input('Carbohydrates (g)', min_value=-1.0, format="%.2f")
                protein = st.number_input('Protein (g)', min_value=-1.0, format="%.2f")
                sugar = st.number_input('Sugar (g)', min_value=-1.0, format="%.2f")
                fiber = st.number_input('Fiber (g)', min_value=-1.0, format="%.2f")
                ean_id = st.text_input('EAN ID')
                submit_button = st.form_submit_button(label='Submit')

            if submit_button:
                # Create a dictionary with the form data
                product_data = {
                    'status': 'buen estado',
                    'days_at_home': 0,
                    'date': init_day(),
                    'product_name': product_name,
                    'generic_name': product,
                    'calories': calories,
                    'fat': fat,
                    'carbs': carbs,
                    'protein': protein,
                    'sugar': sugar,
                    'fiber': fiber,
                    'ean_id': ean_id,
                }

                # Append the dictionary to the list in session state
                st.session_state['data']['food_db'].append(product_data)
                st.success("Product data submitted successfully!")

                # Write the updated list to the JSON file
                write_json_file(json_file_path, st.session_state['data'])

        # Display the stored data
        with col2:
            if st.session_state['data']:
                st.subheader("🧊 Stored Data")
                st.write(st.session_state['data'])




elif section == "Sincronizar Compra":
    # Create two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🔄🛒 Sincronizar Compra")
        if st.button("Submit"):
            # Init Look Up module:
            look_up = FoodLookUp(st.session_state['data'])
            look_up.sync_db(STATUS_LIMIT)
            write_json_file(json_file_path, st.session_state['data'])
            st.success("🔄🧊 Compra sincronizada con exito!")

    with col2:
        if st.session_state['data']:
            st.write(st.session_state['data'])



elif section == "Batch cooking Chef":
    col1, col2 = st.columns([1, 1])
    MENU_CONSTRAINS=[]
    N_DAYS=[]
    id_thread=0
    INPUT_INGREDIENTS = filter_avalaible_ingredients(st.session_state['data'])

    with col1:
        st.subheader("Please select a free combination for your plan.")
        with st.form(key='features_plan'):
            plan_button = st.form_submit_button(label='Submit Plan')
            option1 = st.multiselect(
                "Select Diet Type",
                ["balanced diet", "vegetarian diet", "vegan diet", "gluten-free diet", "low-carb diet", "protein diet"],
                ["balanced diet"],
            )
            st.markdown("***")
            option2 = st.multiselect(
                "Select Flavor Profile",
                ["spicy flavor", "sweet flavor", "savory flavor", "smoky flavor", "umami flavor", "bitter flavor"],
                ["savory flavor"],
            )
            st.markdown("***")
            option3 = st.multiselect(
                "Select Cooking Style",
                ["italian style", "french style", "oriental style", "mediterranean style"],
                ["mediterranean style"],
            )
            st.markdown("***")
            option4 = st.slider("For many days?", min_value=1, max_value=7, value=2, step=1)
            st.markdown("***")
            st.markdown("***")
            option5 = st.multiselect("Would you like to remove any ingredient?", INPUT_INGREDIENTS+["None"], "None")
            st.markdown("***")
            option6 = st.multiselect("Would you prefer to select specific ingredient?", INPUT_INGREDIENTS + ["None"], "None")

        if plan_button:
            MENU_CONSTRAINS = option1 + option2 + option3

            if "None" not in option6:
                INPUT_INGREDIENTS = [i for i in INPUT_INGREDIENTS if i in option6]
            elif len(option5) > 0:
                INPUT_INGREDIENTS = [i for i in INPUT_INGREDIENTS if i not in option5]
            else:
                INPUT_INGREDIENTS = INPUT_INGREDIENTS

            N_DAYS = int(option4)
            id_thread = id_thread + 1

    with col2:
        st.subheader("Batch cooking plan.")
        nodes = Nodes(llm)
        workflow = StateGraph(AgentState)
        workflow = Workflow(llm)
        app = workflow.app
        if plan_button:
            with st.spinner('Wait for it...'):
                thread = {"configurable": {"thread_id": str(id_thread)}}
                for s in app.stream({
                    "N_DAYS": N_DAYS,
                    "INPUT_INGREDIENTS": INPUT_INGREDIENTS,
                    "MENU_CONSTRAINS": MENU_CONSTRAINS,
                    'MAX_REVISIONS': MAX_REVISIONS
                }, thread):
                    print(s)
            st.success("Done!")
            st.markdown(s["translate_menu"]["CURRENT_MENU"], unsafe_allow_html=True)

elif section == "Hazme un plato":

    col1, col2 = st.columns([1, 1])
    INPUT_INGREDIENTS = filter_avalaible_ingredients(st.session_state['data'])

    with col1:
        st.subheader("Please select your preferences for the dish.")
        with st.form(key='features_dish'):
            dish_button = st.form_submit_button(label='Submit Dish')
            input1 = st.multiselect(
                "Select Dish Mood",
                TAG_LIST,
                ['easy', 'vietnamese'],
            )
            st.markdown("***")
            input2 = st.slider("Select number steps", 0, 150, (3, 10))
            st.markdown("***")
            input3 = st.slider("Cook in less than (in minutes)", min_value=0, max_value=360, value=60, step=5)
            st.markdown("***")
            bottom_ingredients = st.slider("Select top-bottom oldest ingredients", min_value=1, max_value=len(INPUT_INGREDIENTS), value=5, step=1)
            st.markdown("***")
            input4 = st.multiselect("Would you like to remove any ingredient?", INPUT_INGREDIENTS+["None"], "None")
            st.markdown("***")
            input5 = st.multiselect("Would you prefer to select specific ingredient?", INPUT_INGREDIENTS+["None"], "None")

        if dish_button:
            if "None" not in input5:
                INPUT_INGREDIENTS = [i for i in INPUT_INGREDIENTS if i in input5]
            elif len(input4) > 0:
                INPUT_INGREDIENTS = [i for i in INPUT_INGREDIENTS if i not in input4]
            else:
                INPUT_INGREDIENTS = INPUT_INGREDIENTS
            INPUT_INGREDIENTS = [i_ingredient for i_ingredient in INPUT_INGREDIENTS if i_ingredient != '']
            INPUT_INGREDIENTS = INPUT_INGREDIENTS[0:bottom_ingredients]

    with col2:
        st.subheader("Daily dish 🍽️.")
        if dish_button:
            # Trigger search engine.
            input_dict = vs.make_dict_inputs(INPUT_INGREDIENTS, input1, input2, input3)
            res = vs.query_and_search(input_dict, use_reranker=True)
            # Trigger LLM dish cook:
            chef = CookAssistant(model_name, api_key)
            output_llm = chef.make_llm_call(input_dict['ingredients'], res)
            st.success("Done!")
            st.markdown(output_llm, unsafe_allow_html=True)