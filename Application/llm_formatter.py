import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class LLMFormatter:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")

        self.client = genai.Client(api_key=api_key)

    def format_recycling_response(self, waste_class: str, retrieved_context: str):
        prompt = f"""
You are a recycling assistant.

Use ONLY the information provided in the CONTEXT below.
Do not add any outside facts, examples, or advice.
If the context is insufficient, say so clearly.

Return JSON with exactly this structure:
{{
  "usefulness": "2-3 sentences explaining why recycling this material is useful",
  "steps": [
    "Step 1: ...",
    "Step 2: ...",
    "Step 3: ...",
    "Step 4: ..."
  ]
}}

MATERIAL:
{waste_class}

CONTEXT:
{retrieved_context}
"""

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                response_mime_type="application/json",
            ),
        )

        return json.loads(response.text)