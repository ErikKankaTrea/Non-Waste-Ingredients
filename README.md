# Non-Waste-Ingredients
IU in streamlit to register and have control of your shopping - each time you shop register it through a barcoder reader or camera device

Powered by AI system - You will be able to get help with the stack of a RAG system + reranking + LLM to make a daily and delicious dish with your oldest ingredients.

FOLLOW ALSO DE IMAGES OF THE APP AS A GUIDE

# 🍽️ **Kitchen Controller App** - **Stop Wasting Food!**

### 📜 **Description:**
Tired of throwing away food from kitchen? Got too many items unused? This is a cool controller app that lets you:
- 📸 Register your shopping using your phone camera or barcode reader.
- 🔄 Keep track of your food items.
- 👨‍🍳 Ask the Chef Assistant to cook a daily dish with your oldest items.
- ➕ Plus, enjoy more features on top!

### ⚙️ **To Make It Yours:**

1. ✏️ **Edit `.env` file** with the following APIs:
   - Twilio
   - Groq
   - Edamam
   - Open Food Facts
   - Ngrok (optional for sharing)

2. 🛒 **Buy a Barcode Reader**

3. 💾 **Install Requirements**:
   - `requirements.txt`
   - fined tuned YOLOv10 find collab notebook ---> I have already trained 70 categories using roboflow for image labeling and YOLOv10 for fine tuning. You should train your own products here the list that its already trained: 
     ['olive', 'avocado', 'garlic', 'apricot', 'celery', 'aquarius', 'eggplant', 'bag of garlic', 'bag of potatoes', 'broccoli', 'zucchini', 'squid', 'carpaccio', 'onion', 'cherry', 'beer', 'star beer', 'sliced ​​mushroom', 'chorizo', 'couscous', 'croissant', 'escalivada', 'spinach', 'noodles', 'flan', 'strawberry', 'gazpacho', 'vegetables', 'egg', 'ham', 'green beans', 'kiwi', 'shrimp', 'lettuce', 'lemon', 'sausage', 'apple', 'peach', 'mint', 'orange', 'bread', 'potato', 'turkey', 'chicken breast', 'pear', 'parsley', 'red pepper', 'green pepper', 'pineapple', 'pistachio', 'pizza', 'banana', 'poke salmon', 'leek', 'cheese', 'sliced cheese', 'bunch of grapes', 'sausages', 'salmon', 'tomato sauce', 'watermelon', 'shushi', 'pork tenderloin', 'special K', 'surimi', 'beef', 'tomato', 'white wine', 'Greek yogurt', 'carrot']

### 🚀 **Getting Started:**

1. 📂 **Create `food_db.json` file**
2. 🛠️ **Modify Path Folders** to fit your system
3. ⚙️ **Run Embeddings File**:
   - **CPU**: ⏳ Takes about 2 hours.
   - **GPU (A100)**: 🚀 Just 20 minutes in Google Colab.

4. 📚 **Make a Sample of the Recipe Book**:
   - 🍳 Process 25% of the recipes (around 2 hours) to insert into the Milvus vector database.

5. 🖥️ **Run Streamlit**:
   - `streamlit run main.py`

6. 📝 **Start Registering Your Shopping**

7. 🔧 **Set Few Params** (if needed)

8. 🍲 **Use the Chef Assistant**:
   - After registering your shopping, let the Chef Assistant create a daily dish based on the oldest products in your kitchen.

9. 🌐 **Use Ngrok to Share** it easily with others.

### 🔜 **To improve:**
- 🍽️ **Batch Cooking Planner**:
   - Make a better planner to assist with batch cooking, focusing on using older products first to minimize waste.
   - Take into account the amount of ingredients. By now it registers the product but only grab carbs, proteins, fat, fiber per 100gr.
