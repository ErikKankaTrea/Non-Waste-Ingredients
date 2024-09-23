import os
from accelerate.commands.config.config import description
from groq import Groq
import json
import re
from dotenv import load_dotenv

class CookAssistant:

    def __init__(self, model_name, api_key):
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def make_llm_call(self, available_ingredients, inputs_prompt: dict):
        if len(inputs_prompt[0])>0:

            steps = re.sub("\[|\]|'", "", inputs_prompt[0]['entity']['steps'])
            time = int(inputs_prompt[0]['entity']['time'])
            description = inputs_prompt[0]['entity']['description']
            ingredients = re.sub("\[|\]|'", "", inputs_prompt[0]['entity']['ingredients'])
            n_steps = inputs_prompt[0]['entity']['n_steps']

            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a professional and helpful cook chef."
                                   f"Using the following inputs, create a recipe using the ingredients and steps provided as a guide to write it."
                                   f"Try to replace the available ingredients with those from the original recipe as much as possible only when it makes sense."
                                   f"Explain the changes made and present the dish inspired by the description provided."

                    },
                    {
                        "role": "user",
                        "content": f"Inputs for the dish:"
                                   f"STEPS: {steps}"
                                   f"INGREDIENTS: {ingredients}."
                                   f"AVAILABLE INGREDIENTS: {available_ingredients}"
                                   f"DESCRIPTION: {description}"
                    }
                ],
                model=self.model_name,
            )

        else:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a professional and helpful cook chef."
                                   f"Using the following input ingredients, create a recipe using them as much as they combine good together."
                                   f"Be precise with timings and amounts for make the recipe"
                                   f"Explain why did you choose this combination of ingredients and present the dish in few nice and cool words."

                    },
                    {
                        "role": "user",
                        "content": f"Inputs for the dish:"
                                   f"INGREDIENTS: {available_ingredients}."
                    }
                ],
                model=self.model_name,
            )
        return chat_completion.choices[0].message.content
