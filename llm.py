import ollama
from openai import OpenAI

from config import *

class LLM:
    def __init__(self, provider):
        self.provider = provider

        if provider == "OpenAI":
            self.client = OpenAI(api_key=OPENAI_API_KEY)
   
    def generate(self,prompt):
        if self.provider == "Ollama":
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}]),
            return response["message"]["content"]
        response = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}]),
        return response.choices[0].message.content
            
