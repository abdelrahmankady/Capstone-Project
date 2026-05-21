import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
client = genai.Client()
for model in client.models.list():
    if "embedContent" in model.supported_actions:
        print(model.name)
