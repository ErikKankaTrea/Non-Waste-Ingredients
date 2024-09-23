from typing_extensions import TypedDict
from typing import List
import os, re, time
from colorama import Fore, Style
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from translate import Translator

class AgentState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        main_menu: Curren menu.
        menu_ingredients: List of ingredients in current menu.
        control_check: List of ingredient that have to be out of the menu.
        revision_number: Number of revision.
        max_revisions: Number of draft trials attempted before stopping.
    """
    ID_STEP: int
    N_DAYS: int
    MENU_CONSTRAINS: List[str]
    INPUT_INGREDIENTS: List[str]
    CURRENT_MENU: str
    MENU_INGREDIENTS: List[str]
    CONTROL_CHECK: bool
    NOT_IN_STOCK: List[str]
    REVISION_NUMBER: int
    MAX_REVISIONS: int


# DRAFT BATCH COOKING MENU:
draft_batch_cooking_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a professional chef specialized in batch cooking. \
You prepare the plans in two parts always: 
The first part is to prepare all the stack of cooked ingredients explained step by step. \
The second part is how to assemble step by step the dishes for every day made from the combinations of prepared ingredients above. 
You must to be precise and explaining all the recipes step by step.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Design in an efficient way, batch cooking plan with following three constrains: \
1.The batch cooking is for the following {N_DAYS}.
2.The batch should be {MENU_CONSTRAINS} \
3.These are the only available ingredients: {INPUT_INGREDIENTS} \
You must be rigorous and concise with the available ingredients selected and do not give other options. \
Be clear and precise describing the steps. \
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

# EXTRACT LIST:
extract_ingredients_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a professional food analyst. Your task is to extract ingredients that are in a given planned menu. \
Return a JSON object with no premable or explaination, the JSON must include a single key 'menu' which must be a list of ingredients that are needed for the given menu. \
Do NOT list kitchen staples such as olive oil, pepper, water, garlic, salt and so on. \
<|eot_id|><|start_header_id|>user<|end_header_id|>
CONTEXT MENU: {CURRENT_MENU} \n\n
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""


# FUZZY COMPARADOR:
ingredients_checker_promt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant that answers with precision and rigor. \
Which elements from list B are not contained in A. Take as contained cases that are very similar/fuzzy, for example: tomato and cherry tomato or lime and limon. \
If all elements are contained say 'OK'. \
Please do NOT count kitchen staples such as olive oil, pepper, water, garlic, salt and so on. \
Return a JSON object with no premable or explaination, the JSON must include a single key 'control' which must be a list. \
<|eot_id|><|start_header_id|>user<|end_header_id|>
A: {INPUT_INGREDIENTS} \
B: {MENU_INGREDIENTS} \
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""


# REWRITE MENU:
modifier_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a practical solver problem kitchen chef. \
You are commited to adjust menus when stock of particular ingredients are gone. \
Correct and modify with common sense following batch cooking plan with additional ingredients given. \
In case ingredients are not similar just remove them from the plan. \
<|eot_id|><|start_header_id|>user<|end_header_id|>
MENU: {CURRENT_MENU} \
OUT OF STOCK: {NOT_IN_STOCK} \
ADDITIONAL INGREDIENTS: {INPUT_INGREDIENTS} \
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

# REFORMAT MENU:
print_menu_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a nice UX designer that format text output. \
Print the presented menu in a format that is easy to read without losing information. \
<|eot_id|><|start_header_id|>user<|end_header_id|>
MENU: {CURRENT_MENU} \
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

# TRANSLATE MENU:
translate_menu_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful english to spanish translator. \
Translate to spanish the following planned batch cooking menu to be read for a spaniard. \
<|eot_id|><|start_header_id|>user<|end_header_id|>
MENU: {CURRENT_MENU} \
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""


class Agents():
    def __init__(self, llm):
        # Generate draft menu:
        draft_menu_prompt = PromptTemplate(
            template=draft_batch_cooking_prompt,
            input_variables=["N_DAYS", "MENU_CONSTRAINS", "INPUT_INGREDIENTS"]
        )
        self.menu_designer = draft_menu_prompt | llm | StrOutputParser()

        # Extract ingredients on menu
        extractor_ingredients_prompt = PromptTemplate(
            template=extract_ingredients_prompt,
            input_variables=["CURRENT_MENU"]
        )
        self.ingredients_extractor = extractor_ingredients_prompt | llm | JsonOutputParser()

        # Comparador de ingredientes
        control_ingredients_prompt = PromptTemplate(
            template=ingredients_checker_promt,
            input_variables=["INPUT_INGREDIENTS", "MENU_LIST"]
        )
        self.control_ingredients = control_ingredients_prompt | llm | JsonOutputParser()

        # Modify current menu
        rewrite_menu_prompt = PromptTemplate(
            template=modifier_prompt,
            input_variables=["CURRENT_MENU", "NOT_IN_STOCK", "INPUT_INGREDIENTS"]
        )
        self.rewrite_menu = rewrite_menu_prompt | llm | StrOutputParser()

        # Re format menu print:
        formatted_menu_prompt = PromptTemplate(
            template=print_menu_prompt,
            input_variables=["CURRENT_MENU"]
        )
        self.reformat_menu = formatted_menu_prompt | llm | StrOutputParser()

        # Translate:
        translated_menu_prompt = PromptTemplate(
            template=translate_menu_prompt,
            input_variables=["CURRENT_MENU"]
        )
        self.translate_menu = translated_menu_prompt | llm | StrOutputParser()



class Nodes:
    def __init__(self, llm):
        self.agents = Agents(llm)

    def plan_menu_node(self, state: AgentState):
        if state['ID_STEP'] is None:
            state['ID_STEP'] = 0
        print(Fore.YELLOW + "Preparing menu...\n" + Style.RESET_ALL)
        response = self.agents.menu_designer.invoke({
            "N_DAYS": state["N_DAYS"],
            "MENU_CONSTRAINS": state["MENU_CONSTRAINS"],
            "INPUT_INGREDIENTS": state["INPUT_INGREDIENTS"]
        })
        return {
                #**state,
                "ID_STEP": state['ID_STEP'],
                "CURRENT_MENU": response}

    def extract_ingredients(self, state: AgentState):
        if state['REVISION_NUMBER'] is None:
            state['REVISION_NUMBER'] = 0
        trials = int(state['REVISION_NUMBER'])
        trials += 1

        print(Fore.MAGENTA + "Extracting ingredients from menu:" + Style.RESET_ALL)
        response = self.agents.ingredients_extractor.invoke({
            "CURRENT_MENU": state["CURRENT_MENU"]
        })
        return {
            # **state,
            "ID_STEP": state['ID_STEP'] + 1,
            "MENU_INGREDIENTS": [item.lower() for item in response['menu']],
            "REVISION_NUMBER": trials
        }


    def eval_ingredients_and_menu(self, state: AgentState):
        print(Fore.MAGENTA + "Eval ingredients from menu:" + Style.RESET_ALL)

        out_stock = [item for item in state["MENU_INGREDIENTS"] if item not in state["INPUT_INGREDIENTS"]]
        control = len(out_stock) == 0
        if control:
            print(Fore.MAGENTA + f"Ingredients that do not match are: {out_stock} - I need to adjust" + Style.RESET_ALL)
        else:
            print(Fore.GREEN + f"All ingredients are available!!" + Style.RESET_ALL)
        return {
            #**state,
            "ID_STEP": state['ID_STEP'] + 1,
            "CONTROL_CHECK": control,
            "NOT_IN_STOCK": out_stock
        }

    def rewrite_menu(self, state: AgentState):
        print(Fore.RED + "Adjusting menu based on available ingredients...\n" + Style.RESET_ALL)
        time.sleep(1)
        response = self.agents.rewrite_menu.invoke({
            "CURRENT_MENU": state["CURRENT_MENU"],
            "NOT_IN_STOCK": state["NOT_IN_STOCK"],
            "INPUT_INGREDIENTS": state["INPUT_INGREDIENTS"]
        })
        return {
            #**state,
            "ID_STEP": state['ID_STEP'] + 1,
            "CURRENT_MENU": response
        }

    def reformat_menu(self, state: AgentState):
        print(Fore.GREEN + "Reformat menu...\n" + Style.RESET_ALL)
        response = self.agents.reformat_menu.invoke({
            "CURRENT_MENU": state["CURRENT_MENU"]
        })
        return {
            #**state,
            "ID_STEP": state['ID_STEP'] + 1,
            "CURRENT_MENU": response
        }

    def translate_menu(self, state: AgentState):
        print(Fore.LIGHTYELLOW_EX + "Translating menu...\n" + Style.RESET_ALL)
        response = self.agents.translate_menu.invoke({
            "CURRENT_MENU": state["CURRENT_MENU"]
        })
        return {
            #**state,
            "ID_STEP": state['ID_STEP'] + 1,
            "CURRENT_MENU": response
        }

    def route_menu_based_on_eval(self, state: AgentState):
        print(Fore.YELLOW + "Routing menu after eval...\n" + Style.RESET_ALL)
        menu_status = state["CONTROL_CHECK"]
        if state["REVISION_NUMBER"] > state["MAX_REVISIONS"]:
            return 'TRANSLATE_MENU'
        elif menu_status:
            return "DONE_MENU"
        else:
            return "ADJUST_MENU"


    #def should_continue(self, state: AgentState):
            #    if state["REVISION_NUMBER"] > state["MAX_REVISIONS"]:
            #return 'reformat_menu'
        #return "eval_menu"




class Workflow():
    def __init__(self, llm):
        # initiate graph state & nodes
        workflow = StateGraph(AgentState)
        nodes = Nodes(llm)

        # define all graph nodes
        workflow.add_node("menu_draft", nodes.plan_menu_node)
        workflow.add_node("extract_ingredients", nodes.extract_ingredients)
        workflow.add_node("eval_menu", nodes.eval_ingredients_and_menu)
        workflow.add_node("rewrite_menu", nodes.rewrite_menu)
        #workflow.add_node("reformat_menu", nodes.reformat_menu)
        workflow.add_node("translate_menu", nodes.translate_menu)

        #workflow.add_conditional_edges(
        #    "extract_ingredients",
        #    nodes.should_continue,
        #    {
        #        END: END,
        #        "eval_menu": "eval_menu"}
        #)

        workflow.add_conditional_edges(
            "eval_menu",
            nodes.route_menu_based_on_eval,
            {
                "TRANSLATE_MENU": "translate_menu",
                "ADJUST_MENU": "rewrite_menu",
                "DONE_MENU": "translate_menu"
            }
        )

        # Connect nodes
        workflow.set_entry_point("menu_draft")
        workflow.add_edge("menu_draft", "extract_ingredients")
        workflow.add_edge("extract_ingredients", "eval_menu")

        workflow.add_edge("rewrite_menu", "extract_ingredients")
        workflow.set_finish_point("translate_menu")

        # Compile
        self.app = workflow.compile()



#MENU_CONSTRAINS = ['low in carbs', 'gluten free']
#INPUT_INGREDIENTS = ['olives', 'avocado', 'broccoli', 'zucchini', 'squid', 'veal carpaccio', 'onion', 'beef',
                     #'tomatoes', 'greek yogurt', 'carrot', 'rice', 'tomato sauce']
#N_DAYS = 4
#MAX_REVISIONS = 1

#thread = {"configurable": {"thread_id": "1"}}
#for s in app.stream({
#    "N_DAYS": N_DAYS,
#    "INPUT_INGREDIENTS": INPUT_INGREDIENTS,
#    "MENU_CONSTRAINS": MENU_CONSTRAINS,
#    'MAX_REVISIONS': MAX_REVISIONS
#}, thread):
#    print(s)


###################              AJUSTAR EL NODO DE EVALUAR, SOBRA EL AGENTE.
###################              LA EXTRACCION DE INGREDIENTES TIENE QUE SER DE LA MATERIA PRIMA.



